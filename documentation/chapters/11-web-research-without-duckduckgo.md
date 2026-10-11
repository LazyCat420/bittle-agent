---
part: Agent tools
status: shipped
updated: 2026-10-06
---

# Web research without DuckDuckGo

`bittle_web_search` (in `app/research_tools.py`) is the agent's web search while it tunes a training run. Until 2026-10-06 it **scraped `lite.duckduckgo.com`**.

## Why it had to go

The workstation and the NAS leave the house through one public IP, and the free search engines bot-block it:

- DuckDuckGo first refused the NAS on 2026-07-27.
- On 2026-10-06 a plain request from the workstation got DuckDuckGo's "bots use DuckDuckGo too" challenge.

The scrape came back empty, and every attempt added to a block that every service in the house shares.

## What replaced it

`shared_web_search()` posts `{query, limit}` to lazy-agent-service's `POST /execute/web_search`. That is Exa's keyless API behind one cache and one rate limit for the whole network; see lazy-agent-service's chapter 04, "Shared keyless web search".

- **Host:** `BITTLE_SEARCH_SERVICE_URL` (falls back to `LAZY_TOOL_SERVICE_URL`, then `http://10.0.0.16:5591`). It is a new `Settings.search_service_url`, also listed in `.env.example`.
- **Results:** `{title, url, snippet}`, as before. Entries without an `http` URL are dropped.
- **A search that did not run** (`rate_limited`, `busy`, `error`) returns `{"ok": false, "error": "search_unavailable", "detail": "<status>: <reason>"}`. The agent can tell "could not search" from "found nothing".
- **The timeout is 30 s.** The service paces Exa calls 1.5 s apart and answers `busy` after 20 s in its queue.
- `BITTLE_ALLOW_WEB_RESEARCH=false` still turns the web tools off.
- **Removed:** the DuckDuckGo lite parser (`_DDGParser`) and its test. arXiv search and page reading are unchanged.

## Verified

- `tests/test_training_tools.py`:
  - `test_web_search_asks_the_shared_search`: exactly one `POST` to `/execute/web_search` with the query and limit, and the result shape;
  - `test_web_search_reports_a_search_that_did_not_run`: a rate-limited search returns `search_unavailable`.
- With the URL pointed back at DuckDuckGo lite, the first test fails.
- `tests/`: 179 passed.


## `/execute` API key (2026-10-11)

`shared_web_search` (`app/research_tools.py`) sends `x-api-key` from `EXECUTE_API_KEY` when set (`631a7c5`); lazy-tool-service now answers 401 without it. The container is stopped on the NAS, so it picks the key up from `.env` on its next deploy. Rollout record and rotation: lazy-agent-service ch.11, Item 3.
