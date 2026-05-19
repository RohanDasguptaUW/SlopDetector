"""Download 2000 real photographs from the Pexels API.

Saves to training/real_photos/<category>_<photo_id>.jpg.
Requires PEXELS_API_KEY environment variable.

Rate limits: 200 requests/hour, 20 000 requests/month.
At 200 req/hr this completes in well under an hour.

Usage:
    PEXELS_API_KEY=<key> python training/download_real_photos.py
"""

import os
import sys
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = "https://api.pexels.com/v1"
OUT_DIR = Path(__file__).parent / "real_photos"
PER_PAGE = 80           # Pexels maximum
INTER_REQUEST_DELAY = 0.5   # seconds between API calls
DOWNLOAD_TIMEOUT = 30
MAX_RETRIES = 5

CATEGORIES: dict[str, int] = {
    "landscape": 200,
    "architecture": 200,
    "street": 200,
    "food": 200,
    "nature": 200,
    "sports": 200,
    "travel": 200,
    "animals": 200,
    "events": 200,
    "candid": 200,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_slug(name: str) -> str:
    return name.replace(" ", "_")


def _retry_get(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
    stream: bool = False,
    timeout: int = 15,
) -> requests.Response:
    """GET with exponential back-off on 429 / 5xx."""
    delay = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, stream=stream, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"    [network error] {exc} — retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
            continue

        if resp.status_code == 429:
            reset = int(resp.headers.get("X-Ratelimit-Reset", time.time() + 60))
            wait = max(1, reset - int(time.time()))
            print(f"    [rate limited] waiting {wait}s …")
            time.sleep(wait)
            continue

        if resp.status_code >= 500:
            if attempt == MAX_RETRIES:
                resp.raise_for_status()
            print(f"    [server error {resp.status_code}] retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
            continue

        return resp

    raise RuntimeError(f"Failed to GET {url} after {MAX_RETRIES} attempts")


def search_photos(
    session: requests.Session,
    query: str,
    page: int,
) -> tuple[list[dict], int]:
    """Returns (photos, total_results)."""
    resp = _retry_get(
        session,
        f"{API_BASE}/search",
        params={
            "query": query,
            "per_page": PER_PAGE,
            "page": page,
            "size": "large",
        },
    )
    resp.raise_for_status()
    time.sleep(INTER_REQUEST_DELAY)
    data = resp.json()
    return data.get("photos", []), data.get("total_results", 0)


def download_image(
    session: requests.Session,
    photo: dict,
    dest: Path,
) -> bool:
    """Download the original-resolution photo. Returns True on success."""
    url = photo["src"]["original"]
    try:
        resp = _retry_get(session, url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        return True
    except Exception as exc:
        print(f"    [download failed] {exc}")
        if dest.exists():
            dest.unlink()
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("PEXELS_API_KEY", "")
    if not api_key:
        sys.exit("Error: PEXELS_API_KEY environment variable not set.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "Authorization": api_key,
    })

    # Resume: collect IDs already on disk
    seen_ids: set[str] = {p.stem.split("_", 1)[-1] for p in OUT_DIR.glob("*.jpg")}
    total_saved = len(seen_ids)
    total_target = sum(CATEGORIES.values())

    print(f"Output directory : {OUT_DIR}")
    print(f"Already present  : {total_saved} photos")
    print(f"Target           : {total_target} photos")
    print()

    for category, target in CATEGORIES.items():
        slug = _safe_slug(category)
        existing = sum(1 for _ in OUT_DIR.glob(f"{slug}_*.jpg"))
        needed = target - existing
        if needed <= 0:
            print(f"[{category}] already complete ({existing}/{target}), skipping.")
            continue

        print(f"[{category}] need {needed} more (have {existing}/{target})")
        saved = 0
        page = 1

        while saved < needed:
            photos, total_results = search_photos(session, category, page)
            if not photos:
                print(f"  [!] no more results at page {page} (total={total_results}), stopping.")
                break

            for photo in photos:
                if saved >= needed:
                    break

                photo_id = str(photo["id"])
                if photo_id in seen_ids:
                    continue

                dest = OUT_DIR / f"{slug}_{photo_id}.jpg"
                photographer = photo.get("photographer", "unknown")
                ok = download_image(session, photo, dest)

                if ok:
                    seen_ids.add(photo_id)
                    saved += 1
                    total_saved += 1
                    pct = total_saved / total_target * 100
                    print(
                        f"  [{total_saved:4d}/{total_target}] {pct:5.1f}%  "
                        f"{dest.name}  (by {photographer})"
                    )

            max_page = -(-total_results // PER_PAGE)  # ceiling division
            if page >= max_page:
                print(f"  [!] exhausted all {total_results} results for '{category}'.")
                break
            page += 1

        print(f"  → {slug}: saved {saved} new photos.\n")

    final = sum(1 for _ in OUT_DIR.glob("*.jpg"))
    print(f"Done. {final} total photos in {OUT_DIR}")


if __name__ == "__main__":
    main()
