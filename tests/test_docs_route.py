"""The documentation page is served by the app itself (a doc nobody can open is a doc nobody corrects)."""

from fastapi.testclient import TestClient

from app.main import DOCS_DIR, app


def test_documentation_is_served_with_media():
    with TestClient(app) as c:
        r = c.get("/documentation", follow_redirects=False)
        assert r.status_code == 307 and r.headers["location"] == "/documentation/"
        page = c.get("/documentation/")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert "Bittle" in page.text and "rl-walking-training" in page.text
        gif = next((DOCS_DIR / "media" / "rl-training").glob("*.gif"))
        m = c.get(f"/documentation/media/rl-training/{gif.name}")
        assert m.status_code == 200 and m.headers["content-type"].startswith("image/gif")
