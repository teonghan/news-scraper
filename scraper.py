#!/usr/bin/env python3
"""
scraper.py — Malaysian News Scraper (CLI / cron-friendly)

Usage examples:
  python scraper.py
  python scraper.py --sources nst malay_mail --delay 2 --max-items 50
  python scraper.py --output-dir ./data --json --csv
  python scraper.py --sources nst --no-summary --include-content
  python scraper.py --quiet   # suppress all non-error output (good for cron)
  python scraper.py --list-sources

Cron example (every 6 hours, output to /data/news):
  0 */6 * * * /usr/bin/python3 /opt/scraper.py --output-dir /data/news --quiet >> /var/log/scraper.log 2>&1
"""

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import StringIO
from pathlib import Path
from time import sleep
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
#  Feed registry  (add more sources here)
# ─────────────────────────────────────────────
FEED_REGISTRY: dict[str, dict] = {
    "nst": {
        "name": "New Straits Times",
        "url": "https://www.nst.com.my/feed",
        # URL path substrings to skip — any article whose link contains one
        # of these will be dropped before fetching its body.
        "exclude_paths": ["/lifestyle", "/sports"],
    },
    "malay_mail": {
        "name": "Malay Mail",
        "url": "https://www.malaymail.com/feed/rss/malaysia",
        "exclude_paths": [],
    },
    # Add more feeds here:
    # "freemalaysiatoday": {
    #     "name": "Free Malaysia Today",
    #     "url": "https://www.freemalaysiatoday.com/feed/",
    #     "exclude_paths": [],
    # },
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/89.0",
]

NAMESPACES = {"content": "http://purl.org/rss/1.0/modules/content/"}

_TRACKING_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "igshid", "mibextid",
}

MALAY_STOPWORDS = {
    "yang", "dan", "atau", "untuk", "kepada", "dalam", "dengan",
    "itu", "ini", "tidak", "ada", "bagi", "oleh", "terhadap",
    "akan", "kerana", "juga",
}

# ─────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────

def setup_logging(quiet: bool, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("scraper")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    if quiet:
        logger.setLevel(logging.WARNING)
    elif verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger


# ─────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────

def random_ua() -> str:
    return random.choice(USER_AGENTS)


def clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = str(s).replace("\xa0", " ").replace("\u200b", " ")
    return re.sub(r"\s+", " ", s).strip()


def canonical_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parts = urlsplit(url)
        q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
             if k not in _TRACKING_KEYS]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q, doseq=True), ""))
    except Exception:
        return url


def safe_iso(pubdate: str | None) -> str | None:
    try:
        if not pubdate or pubdate == "No date":
            return None
        dt = parsedate_to_datetime(pubdate)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def make_id(title: str | None, link: str | None) -> str:
    base = f"{title or ''}|{link or ''}"
    return hashlib.sha1(base.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────
#  NLTK summariser
# ─────────────────────────────────────────────

def _ensure_nltk(logger: logging.Logger) -> None:
    import nltk
    needed = [("tokenizers/punkt", "punkt"), ("corpora/stopwords", "stopwords")]
    for path, pkg in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.debug("Downloading NLTK resource: %s", pkg)
            nltk.download(pkg, quiet=True)


def summarize(text: str, max_chars: int = 600, max_sentences: int = 3,
              logger: logging.Logger | None = None) -> str:
    text = clean_text(text)
    if not text:
        return text
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import sent_tokenize, word_tokenize

        if logger:
            _ensure_nltk(logger)

        sw = (set(stopwords.words("english"))
              | set(stopwords.words("indonesian"))
              | MALAY_STOPWORDS)

        sentences = sent_tokenize(text)
        words = [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in sw]
        if not sentences or not words:
            raise ValueError("Empty after tokenization")

        freq = Counter(words)
        scored = []
        for i, s in enumerate(sentences):
            toks = [w for w in word_tokenize(s.lower()) if w.isalpha()]
            sc = sum(freq.get(w, 0) for w in toks) / (1 + len(toks))
            sc *= 1.05 ** max(0, (len(sentences) - i))
            scored.append((i, sc))

        keep = sorted(i for i, _ in
                      sorted(scored, key=lambda x: x[1], reverse=True)
                      [:max(1, min(max_sentences, len(sentences)))])
        summary = clean_text(" ".join(sentences[i] for i in keep))
        return summary
    except Exception:
        return (text[:max_chars].rsplit(" ", 1)[0] + "…") if len(text) > max_chars else text


# ─────────────────────────────────────────────
#  Scraping core
# ─────────────────────────────────────────────

def fetch_rss(url: str, logger: logging.Logger) -> ET.Element | None:
    headers = {"User-Agent": random_ua()}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError) as e:
        logger.error("RSS fetch/parse failed: %s — %s", url, e)
        return None



def fetch_body(url: str, logger: logging.Logger) -> str:
    headers = {"User-Agent": random_ua()}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        node = (
            soup.find("article")
            or soup.find("div", class_="entry-content")
            or soup.find("div", class_="td-post-content")
            or soup.find("div", class_="g-item-content")
        )
        if node:
            for s in node(["script", "style"]):
                s.extract()
            return node.get_text(separator="\n", strip=True)
        return "Content not available (specific article tag not found)"
    except requests.RequestException as e:
        logger.warning("Article fetch error: %s — %s", url, e)
        return "Content not available (network/request error)"
    except Exception as e:
        logger.warning("Article parse error: %s — %s", url, e)
        return "Content not available (parsing error)"


def parse_article(item: ET.Element, link: str, logger: logging.Logger) -> str:
    """Mirror original logic: prefer content:encoded, then description, then scrape."""
    content_encoded = item.find("content:encoded", NAMESPACES)
    description_el = item.find("description")
    description = description_el.text if description_el is not None else ""

    if content_encoded is not None and content_encoded.text and content_encoded.text.strip():
        soup = BeautifulSoup(content_encoded.text, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    if description:
        soup = BeautifulSoup(description, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    return fetch_body(link, logger)

def scrape_feed(key: str, meta: dict, args: argparse.Namespace,
                logger: logging.Logger) -> list[dict]:
    feed_name = meta["name"]
    feed_url = meta["url"]
    rows: list[dict] = []

    logger.info("Fetching feed: %s (%s)", feed_name, feed_url)
    feed = fetch_rss(feed_url, logger)
    if not feed:
        return rows

    items = feed.findall(".//item")
    logger.info("  Found %d articles in '%s'", len(items), feed_name)

    # Build the combined exclude list: registry defaults + CLI overrides
    registry_excludes = meta.get("exclude_paths", [])
    cli_excludes = args.exclude_paths or []
    all_excludes = [p.lower() for p in registry_excludes + cli_excludes]

    max_fetch = args.max_fetch if args.max_fetch and args.max_fetch > 0 else len(items)

    skipped = 0
    for i, item in enumerate(items, 1):
        title = item.find("title")
        link_el = item.find("link")
        pubdate = item.find("pubDate")

        title_text = title.text if title is not None else "No title"
        link_text = link_el.text if link_el is not None else "No link"
        date_text = pubdate.text if pubdate is not None else "No date"

        # ── URL-path filter ──────────────────────────────────────────
        link_lower = link_text.lower()
        if any(excl in link_lower for excl in all_excludes):
            logger.debug("  [skip] %s", link_text)
            skipped += 1
            continue
        # ─────────────────────────────────────────────────────────────

        article = parse_article(item, link_text, logger)
        rows.append({
            "Source": feed_name,
            "Title": title_text,
            "Link": link_text,
            "Article": article,
            "Date": date_text,
        })
        logger.debug("  [%d/%d] %s", i, len(items), title_text[:80])

        if len(rows) >= max_fetch:
            logger.info("  Reached --max-fetch limit (%d), stopping early.", max_fetch)
            break

        if i < len(items):
            sleep(args.delay)

    if skipped:
        logger.info("  Skipped %d article(s) matching exclude_paths.", skipped)

    return rows


# ─────────────────────────────────────────────
#  Output builders
# ─────────────────────────────────────────────

def build_json(articles: list[dict], args: argparse.Namespace,
               logger: logging.Logger) -> dict:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    items: list[dict] = []
    seen: set[str] = set()

    for a in articles:
        title = clean_text(a.get("Title"))
        url = canonical_url(a.get("Link"))
        uid = make_id(title, url)
        if uid in seen:
            continue
        seen.add(uid)

        summary_text = "" if args.no_summary else summarize(
            clean_text(a.get("Article", "")),
            max_chars=args.summary_chars,
            max_sentences=args.summary_sentences,
            logger=logger,
        )

        item: dict = {
            "id": uid,
            "title": title,
            "url": url,
            "source": clean_text(a.get("Source")),
            "publishedAt": safe_iso(a.get("Date")),
            "summary": summary_text,
        }
        if args.include_content:
            item["content"] = clean_text(a.get("Article"))
        items.append(item)

    items = sorted(items, key=lambda x: x["publishedAt"] or "", reverse=True)
    items = items[:args.max_items]
    return {"asOf": now_iso, "totalItems": len(items), "items": items}


def build_csv(articles: list[dict]) -> str:
    import csv
    buf = StringIO()
    fieldnames = ["Source", "Title", "Link", "Date", "Article"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(articles)
    return buf.getvalue()


# ─────────────────────────────────────────────
#  Argument parser
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    all_keys = list(FEED_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        prog="scraper.py",
        description="Malaysian News Scraper — CLI / cron-friendly",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source selection
    parser.add_argument(
        "--sources", nargs="+", metavar="SOURCE",
        default=all_keys,
        choices=all_keys,
        help=f"Feed keys to scrape. Available: {', '.join(all_keys)}",
    )
    parser.add_argument(
        "--list-sources", action="store_true",
        help="Print available source keys and exit.",
    )
    parser.add_argument(
        "--exclude-paths", nargs="+", metavar="PATH", default=[],
        help=(
            "Extra URL path substrings to skip for ALL sources, "
            "e.g. --exclude-paths /opinion /video. "
            "Per-source exclusions are defined in FEED_REGISTRY."
        ),
    )

    # Output
    parser.add_argument(
        "--output-dir", metavar="DIR", default=".",
        help="Directory to write output files.",
    )
    parser.add_argument(
        "--json", dest="write_json", action="store_true", default=True,
        help="Write JSON output (default: on).",
    )
    parser.add_argument(
        "--no-json", dest="write_json", action="store_false",
        help="Disable JSON output.",
    )
    parser.add_argument(
        "--csv", dest="write_csv", action="store_true", default=False,
        help="Write CSV output.",
    )
    parser.add_argument(
        "--no-csv", dest="write_csv", action="store_false",
        help="Disable CSV output (default: off).",
    )
    parser.add_argument(
        "--filename-prefix", metavar="PREFIX", default="news",
        help="Prefix for output filenames, e.g. 'news' → news_20240101T120000Z.json",
    )
    parser.add_argument(
        "--no-timestamp", action="store_true", default=False,
        help="Use fixed filenames (latest_news.json) instead of timestamped ones.",
    )

    # Scraping behaviour
    parser.add_argument(
        "--delay", type=float, default=2.0, metavar="SEC",
        help="Seconds to wait between article fetches.",
    )

    # JSON content options
    parser.add_argument(
        "--max-items", type=int, default=80, metavar="N",
        help="Maximum articles in JSON output (applied after scraping).",
    )
    parser.add_argument(
        "--max-fetch", type=int, default=0, metavar="N",
        help="Stop fetching after N articles per feed (0 = no limit). "
             "Limits actual HTTP requests, unlike --max-items which only "
             "trims the final JSON.",
    )
    parser.add_argument(
        "--include-content", action="store_true", default=False,
        help="Include full article text in JSON (makes file larger).",
    )
    parser.add_argument(
        "--no-summary", action="store_true", default=False,
        help="Omit the NLTK summary field from JSON.",
    )
    parser.add_argument(
        "--summary-chars", type=int, default=600, metavar="N",
        help="Max characters per summary (only applies to the except/fallback path).",
    )
    parser.add_argument(
        "--summary-sentences", type=int, default=3, metavar="N",
        help="Max sentences per summary.",
    )

    # Verbosity
    parser.add_argument(
        "--quiet", "-q", action="store_true", default=False,
        help="Suppress info output (warnings/errors only). Ideal for cron.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Enable debug-level output.",
    )

    return parser


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logger = setup_logging(args.quiet, args.verbose)

    # --list-sources shortcut
    if args.list_sources:
        print("\nAvailable sources:")
        for key, meta in FEED_REGISTRY.items():
            print(f"  {key:<20} {meta['name']}")
            print(f"  {'':20} {meta['url']}")
        return 0

    # Validate output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp for filenames
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    def make_path(ext: str) -> Path:
        if args.no_timestamp:
            return out_dir / f"latest_{args.filename_prefix}.{ext}"
        return out_dir / f"{args.filename_prefix}_{ts}.{ext}"

    # Scrape all selected feeds
    all_articles: list[dict] = []
    sources = {k: FEED_REGISTRY[k] for k in args.sources}

    for i, (key, meta) in enumerate(sources.items(), 1):
        logger.info("── Feed %d/%d: %s", i, len(sources), meta["name"])
        rows = scrape_feed(key, meta, args, logger)
        all_articles.extend(rows)
        if i < len(sources):
            sleep(0.5)

    if not all_articles:
        logger.warning("No articles scraped. Exiting.")
        return 1

    logger.info("Total raw articles: %d", len(all_articles))

    # Write outputs
    written: list[Path] = []

    if args.write_json:
        payload = build_json(all_articles, args, logger)
        path = make_path("json")
        path.write_bytes(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        logger.info("JSON written → %s  (%d items)", path, payload["totalItems"])
        written.append(path)

    if args.write_csv:
        path = make_path("csv")
        path.write_text(build_csv(all_articles), encoding="utf-8-sig")
        logger.info("CSV  written → %s  (%d rows)", path, len(all_articles))
        written.append(path)

    if not written:
        logger.warning("No output format selected (use --json / --csv).")

    return 0


if __name__ == "__main__":
    sys.exit(main())