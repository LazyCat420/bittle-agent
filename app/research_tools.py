"""Web research tools for the GLM harness.

Lets the agent look things up while it tunes a training run: web search
(DuckDuckGo lite, keyless, ~1 s), arXiv paper search, page reading, and a
small persistent notebook under ``storage/research/`` so findings survive
across sessions. Outbound fetches are guarded against private/LAN targets.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import time
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx

RESEARCH_TOOL_NAMES = (
    "bittle_web_search",
    "bittle_search_papers",
    "bittle_read_url",
    "bittle_save_research_note",
    "bittle_list_research_notes",
    "bittle_read_research_note",
)

RESEARCH_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "bittle_web_search",
        "description": "Search the web (DuckDuckGo). Use it to find how others tuned quadruped RL (reward weights, PPO settings, domain randomisation), sim-to-real reports for Petoi Bittle / OpenCat, MuJoCo Warp or Brax issues, etc.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer", "description": "1-10, default 6"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "bittle_search_papers",
        "description": "Search arXiv for research papers (title, abstract excerpt, link). Good for 'learning quadruped locomotion small robot sim2real', 'PPO reward shaping legged', 'domain randomization servo latency'.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer", "description": "1-10, default 5"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "bittle_read_url",
        "description": "Fetch a public web page (or GitHub file) as plain text, truncated. Use after a search to read details. PDFs are not extracted; prefer arXiv abstract pages or HTML versions.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}, "max_chars": {"type": "integer", "description": "500-20000, default 6000"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "bittle_save_research_note",
        "description": "Save a research finding (what was tried / what others report / a hypothesis) to the persistent notebook so future sessions can reuse it.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "content": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "sources": {"type": "array", "items": {"type": "string"}, "description": "URLs"}},
            "required": ["title", "content"]}}},
    {"type": "function", "function": {
        "name": "bittle_list_research_notes",
        "description": "List saved research notes (title, tags, excerpt).",
        "parameters": {"type": "object", "properties": {"tag": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "bittle_read_research_note",
        "description": "Read one saved research note in full by its name.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
]

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Safari/537.36")
MAX_BODY_BYTES = 2_000_000
NOTES_DIR = Path(__file__).resolve().parent.parent / "storage" / "research"


# ── safety: never let the agent point the NAS at the LAN ───────────────────

def _is_public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False
    return bool(infos)


def check_url(url: str) -> str | None:
    """Return an error string if the URL must not be fetched."""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return "only http(s) URLs are allowed"
    if not p.hostname:
        return "URL has no host"
    if p.hostname in ("localhost",) or not _is_public_host(p.hostname):
        return "refusing to fetch a private/LAN address"
    return None


# ── HTML → text ────────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "pre", "section", "article", "td", "th"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        t = "".join(self.parts)
        t = re.sub(r"[ \t\r\f\v]+", " ", t)
        t = re.sub(r"\n\s*\n+", "\n\n", t)
        return t.strip()


def html_to_text(html: str) -> tuple[str, str]:
    ex = _TextExtractor()
    try:
        ex.feed(html)
    except Exception:
        pass
    return ex.title.strip(), ex.text()


# ── DuckDuckGo lite parser ─────────────────────────────────────────────────

class _DDGParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._cur: dict[str, str] | None = None
        self._in_link = False
        self._in_snippet = False
        self._row_sponsored = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "") or ""
        if tag == "tr":
            self._row_sponsored = "sponsored" in cls
        if tag == "a" and "result-link" in cls and not self._row_sponsored:
            href = a.get("href", "") or ""
            if href.startswith("//"):
                href = "https:" + href
            # DDG wraps organic results as /l/?uddg=<target> (relative or absolute)
            pu = urlparse(href)
            if pu.path == "/l/" and "uddg" in parse_qs(pu.query):
                href = parse_qs(pu.query)["uddg"][0]
            self._cur = {"title": "", "url": href, "snippet": ""}
            self._in_link = True
        if tag == "td" and "result-snippet" in cls and self.results:
            self._in_snippet = True

    def handle_endtag(self, tag):
        if tag == "a" and self._in_link and self._cur is not None:
            self._cur["title"] = self._cur["title"].strip()
            self.results.append(self._cur)
            self._cur = None
            self._in_link = False
        if tag == "td" and self._in_snippet:
            self._in_snippet = False

    def handle_data(self, data):
        if self._in_link and self._cur is not None:
            self._cur["title"] += data
        elif self._in_snippet and self.results:
            self.results[-1]["snippet"] = (self.results[-1]["snippet"] + data).strip()


async def ddg_search(query: str, limit: int = 6, *, timeout: float = 12.0) -> list[dict[str, str]]:
    url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    parser = _DDGParser()
    parser.feed(resp.text)
    seen, out = set(), []
    for r in parser.results:
        if r["url"] in seen or not r["url"].startswith("http"):
            continue
        seen.add(r["url"])
        out.append(r)
        if len(out) >= limit:
            break
    return out


# ── arXiv ──────────────────────────────────────────────────────────────────

ATOM = "{http://www.w3.org/2005/Atom}"


async def arxiv_search(query: str, limit: int = 5, *, timeout: float = 15.0) -> list[dict[str, Any]]:
    url = (f"https://export.arxiv.org/api/query?search_query=all:{quote_plus(query)}"
           f"&start=0&max_results={limit}&sortBy=relevance&sortOrder=descending")
    async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    root = ET.fromstring(resp.text)
    out = []
    for e in root.findall(f"{ATOM}entry"):
        title = re.sub(r"\s+", " ", (e.findtext(f"{ATOM}title") or "")).strip()
        summary = re.sub(r"\s+", " ", (e.findtext(f"{ATOM}summary") or "")).strip()
        authors = [a.findtext(f"{ATOM}name") or "" for a in e.findall(f"{ATOM}author")]
        pdf = next((l.get("href") for l in e.findall(f"{ATOM}link") if l.get("title") == "pdf"), None)
        out.append({"title": title, "authors": authors[:5], "published": (e.findtext(f"{ATOM}published") or "")[:10],
                    "url": e.findtext(f"{ATOM}id"), "pdf": pdf, "abstract": summary[:700]})
    return out


# ── page reader ────────────────────────────────────────────────────────────

def _rewrite_url(url: str) -> str:
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)$", url)
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
    m = re.match(r"https?://github\.com/([^/]+)/([^/#?]+)/?$", url)
    if m:  # repo root -> its README, not the site chrome
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/HEAD/README.md"
    m = re.match(r"https?://arxiv\.org/pdf/(\d+\.\d+)(v\d+)?(\.pdf)?$", url)
    if m:
        return f"https://arxiv.org/abs/{m.group(1)}"
    return url


async def read_url(url: str, max_chars: int = 6000, *, timeout: float = 20.0) -> dict[str, Any]:
    url = _rewrite_url(url.strip())
    err = check_url(url)
    if err:
        return {"ok": False, "error": "blocked", "detail": err, "url": url}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        async with client.stream("GET", url) as resp:
            ctype = resp.headers.get("content-type", "")
            chunks, size = [], 0
            async for chunk in resp.aiter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size >= MAX_BODY_BYTES:
                    break
            status = resp.status_code
    body = b"".join(chunks)
    if status >= 400:
        return {"ok": False, "error": f"http_{status}", "url": url}
    if "pdf" in ctype.lower() or body[:5] == b"%PDF-":
        return {"ok": False, "error": "pdf_not_supported", "url": url,
                "detail": "PDF bodies are not extracted; use the abstract/HTML page instead"}
    text = body.decode(resp.encoding or "utf-8", errors="replace")
    if "html" in ctype.lower() or text.lstrip()[:1] == "<":
        title, text = html_to_text(text)
    else:
        title = ""
    max_chars = int(min(max(max_chars, 500), 20000))
    truncated = len(text) > max_chars
    return {"ok": True, "url": url, "title": title, "content_type": ctype.split(";")[0],
            "chars": len(text), "truncated": truncated, "text": text[:max_chars]}


# ── notebook ───────────────────────────────────────────────────────────────

def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:60] or f"note-{int(time.time())}"


class ResearchTools:
    def __init__(self, settings, notes_dir: Path = NOTES_DIR):
        self.settings = settings
        self.notes_dir = notes_dir

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name in ("bittle_web_search", "bittle_search_papers", "bittle_read_url") and not getattr(self.settings, "allow_web_research", True):
            return {"ok": False, "error": "web_research_disabled", "detail": "BITTLE_ALLOW_WEB_RESEARCH=false"}
        try:
            return await getattr(self, "_" + name.removeprefix("bittle_"))(args)
        except httpx.HTTPError as exc:
            return {"ok": False, "error": "network_error", "detail": str(exc)[:300]}
        except Exception as exc:  # research must never crash the harness
            return {"ok": False, "error": "research_error", "detail": str(exc)[:300]}

    async def _web_search(self, args):
        limit = int(min(max(args.get("limit", 6) or 6, 1), 10))
        results = await ddg_search(str(args["query"]), limit)
        return {"ok": True, "query": args["query"], "results": results, "count": len(results)}

    async def _search_papers(self, args):
        limit = int(min(max(args.get("limit", 5) or 5, 1), 10))
        papers = await arxiv_search(str(args["query"]), limit)
        return {"ok": True, "query": args["query"], "papers": papers, "count": len(papers)}

    async def _read_url(self, args):
        return await read_url(str(args["url"]), int(args.get("max_chars", 6000) or 6000))

    async def _save_research_note(self, args):
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        name = _slug(str(args["title"]))
        note = {"name": name, "title": str(args["title"]), "content": str(args["content"])[:20000],
                "tags": [str(t) for t in (args.get("tags") or [])][:10],
                "sources": [str(s) for s in (args.get("sources") or [])][:20],
                "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (self.notes_dir / f"{name}.json").write_text(json.dumps(note, indent=2))
        return {"ok": True, "name": name, "path": str(self.notes_dir / f"{name}.json")}

    async def _list_research_notes(self, args):
        tag = args.get("tag")
        notes = []
        if self.notes_dir.is_dir():
            for p in sorted(self.notes_dir.glob("*.json")):
                try:
                    n = json.loads(p.read_text())
                except Exception:
                    continue
                if tag and tag not in n.get("tags", []):
                    continue
                notes.append({"name": n["name"], "title": n["title"], "tags": n.get("tags", []),
                              "saved": n.get("saved"), "excerpt": n["content"][:160]})
        return {"ok": True, "notes": notes, "count": len(notes)}

    async def _read_research_note(self, args):
        p = self.notes_dir / f"{_slug(str(args['name']))}.json"
        if not p.is_file():
            return {"ok": False, "error": "not_found", "detail": args.get("name")}
        return {"ok": True, "note": json.loads(p.read_text())}
