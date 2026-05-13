"""Download 2000 real photographs from Unsplash with intact EXIF data.

Saves to training/real_photos/ as JPEGs named <category>_<photo_id>.jpg.
Requires UNSPLASH_ACCESS_KEY environment variable.

NOTE on rate limits:
    Demo apps:        50  requests / hour  (~40 h for 2000 photos)
    Production apps:  5000 requests / hour (~25 min for 2000 photos)
    Apply for production access at https://unsplash.com/developers

Usage:
    UNSPLASH_ACCESS_KEY=<key> python training/download_real_photos.py
"""

import os
import sys
import time
import hashlib
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = "https://api.unsplash.com"
OUT_DIR = Path(__file__).parent / "real_photos"
PER_PAGE = 30           # Unsplash max per search page
INTER_REQUEST_DELAY = 0.5   # seconds between API calls — be polite
DOWNLOAD_TIMEOUT = 30       # seconds for image download
MAX_RETRIES = 5

# 2000 photos split evenly across 5 subjects
CATEGORIES: dict[str, int] = {
    "portrait": 400,
    "landscape": 400,
    "street photography": 400,
    "events": 400,
    "sports": 400,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_slug(name: str) -> str:
    return name.replace(" ", "_")


def _retry_get(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    stream: bool = False,
    timeout: int = 15,
) -> requests.Response:
    """GET with exponential back-off on 429 / 5xx."""
    delay = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, headers=headers,
                               stream=stream, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"    [network error] {exc} — retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
            continue

        if resp.status_code == 429:
            wait = int(resp.headers.get("X-Ratelimit-Reset", delay + 60))
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
    headers: dict,
    query: str,
    page: int,
) -> list[dict]:
    resp = _retry_get(
        session, f"{API_BASE}/search/photos",
        params={
            "query": query,
            "per_page": PER_PAGE,
            "page": page,
            "orientation": "squarish",
            "content_filter": "high",
        },
        headers=headers,
    )
    resp.raise_for_status()
    time.sleep(INTER_REQUEST_DELAY)
    return resp.json().get("results", [])


def trigger_download(
    session: requests.Session,
    headers: dict,
    photo_id: str,
) -> None:
    """
    Required by Unsplash API guidelines whenever an image is downloaded.
    https://help.unsplash.com/en/articles/2511258-guideline-triggering-a-download
    """
    try:
        _retry_get(session, f"{API_BASE}/photos/{photo_id}/download",
                   headers=headers)
        time.sleep(INTER_REQUEST_DELAY)
    except Exception:
        pass  # non-fatal — best effort


def download_image(
    session: requests.Session,
    raw_url: str,
    dest: Path,
) -> bool:
    """Download raw Unsplash URL as JPEG. Returns True on success."""
    # ?fm=jpg preserves EXIF; q=90 keeps quality high
    url = raw_url.split("?")[0] + "?fm=jpg&q=90&fit=max&w=2000"
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
    access_key = os.environ.get("UNSPLASH_ACCESS_KEY", "")
    if not access_key:
        sys.exit("Error: UNSPLASH_ACCESS_KEY environment variable not set.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"Client-ID {access_key}"}
    session = requests.Session()
    session.headers.update({"Accept-Version": "v1"})

    # Track seen IDs globally to avoid cross-category duplicates
    seen_ids: set[str] = {p.stem.split("_", 1)[-1] for p in OUT_DIR.glob("*.jpg")}
    total_saved = len(seen_ids)
    total_target = sum(CATEGORIES.values())

    print(f"Output directory : {OUT_DIR}")
    print(f"Already present  : {total_saved} photos")
    print(f"Target           : {total_target} photos")
    print()

    for category, target in CATEGORIES.items():
        slug = _safe_slug(category)
        existing = sum(1 for p in OUT_DIR.glob(f"{slug}_*.jpg"))
        needed = target - existing
        if needed <= 0:
            print(f"[{category}] already complete ({existing}/{target}), skipping.")
            continue

        print(f"[{category}] need {needed} more (have {existing}/{target})")
        saved = 0
        page = 1

        while saved < needed:
            photos = search_photos(session, headers, category, page)
            if not photos:
                print(f"  [!] no more results at page {page}, stopping.")
                break

            for photo in photos:
                if saved >= needed:
                    break

                photo_id = photo["id"]
                if photo_id in seen_ids:
                    continue

                # Skip photos without EXIF camera data (per photo metadata)
                exif = photo.get("exif") or {}
                if not exif.get("make") and not exif.get("model"):
                    continue

                dest = OUT_DIR / f"{slug}_{photo_id}.jpg"
                raw_url = photo["urls"]["raw"]

                trigger_download(session, headers, photo_id)
                ok = download_image(session, raw_url, dest)

                if ok:
                    seen_ids.add(photo_id)
                    saved += 1
                    total_saved += 1
                    pct = total_saved / total_target * 100
                    camera = f"{exif.get('make', '')} {exif.get('model', '')}".strip()
                    print(
                        f"  [{total_saved:4d}/{total_target}] {pct:5.1f}%  "
                        f"{slug}_{photo_id}.jpg  ({camera})"
                    )

            page += 1

        print(f"  → {slug}: saved {saved} new photos.\n")

    final = sum(1 for _ in OUT_DIR.glob("*.jpg"))
    print(f"Done. {final} total photos in {OUT_DIR}")


if __name__ == "__main__":
    main()
