#!/usr/bin/env python3
"""
3D Printed Spare Parts — Automated Content Pipeline
====================================================

HOW IT WORKS (high-level):
  1. NewsAPI      → Fetch recent English news about "3D printed spare parts"
  2. Claude AI    → Score relevance, rewrite in professional tone, assign
                    a category, and generate SEO tags  (structured output)
  3. WordPress.com → Publish any article that scores 7 or higher

DUPLICATE PREVENTION:
  Every article URL we process (pass or fail the score threshold) is saved
  to posted_articles.json so it's never analysed or posted again.

LOGGING:
  Everything is written to pipeline.log AND printed to the terminal.

SCHEDULING:
  This script is meant to run daily at 08:00 UTC.
  Railway handles that via the cron in railway.toml.

REQUIREMENTS:
  pip install -r requirements.txt
  Fill in all values in your .env file before running.
"""

from __future__ import annotations   # Allows type hints like list[str] on Python < 3.10

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD-LIBRARY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY IMPORTS  (installed via requirements.txt)
# ─────────────────────────────────────────────────────────────────────────────
import requests                   # HTTP calls to NewsAPI and WordPress.com
from anthropic import Anthropic   # Official Claude SDK
from dotenv import load_dotenv    # Reads .env into os.environ at startup
from pydantic import BaseModel    # Data-validation for Claude's structured output


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# Load .env FIRST so every os.getenv() call below finds the values.
load_dotenv(override=True)   # override=True ensures .env values win even if the
                            # variable already exists as an empty string in the OS env

# ── File paths ────────────────────────────────────────────────────────────────
LOG_FILE            = "pipeline.log"          # Appended on every run
POSTED_TRACKER_FILE = "posted_articles.json"  # Tracks processed article URLs

# ── API credentials  (never hard-code these — they live in .env) ─────────────
NEWSAPI_KEY   = os.getenv("NEWSAPI_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")   # Read automatically by Anthropic()
WP_SITE_ID    = os.getenv("WP_SITE_ID")           # e.g. "3dprintedspareparts.com"
WP_TOKEN      = os.getenv("WP_TOKEN")             # WordPress.com Bearer token

# ── Pipeline settings — feel free to tweak these ─────────────────────────────
NEWS_QUERY          = "3D printed spare parts"
NEWS_PAGE_SIZE      = 10     # How many articles to pull per run (max 100 on free NewsAPI)
RELEVANCE_THRESHOLD = 7      # Minimum score out of 10 needed to publish
CLAUDE_MODEL        = "claude-opus-4-6"
DELAY_BETWEEN_CALLS = 1.5    # Seconds between Claude calls — respects rate limits


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
# We write every log message to BOTH a file (for Railway's log viewer) and
# to the terminal (handy when running locally).

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),   # → pipeline.log
        logging.StreamHandler(),                            # → terminal
    ],
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DUPLICATE TRACKER
# ═════════════════════════════════════════════════════════════════════════════

def load_posted_urls() -> set[str]:
    """
    Load the set of article URLs we have already processed from disk.

    Why URLs?  Each NewsAPI article has a unique canonical URL — a reliable
    identifier that won't change between pipeline runs.

    Returns an empty set on the very first run (file doesn't exist yet).
    """
    path = Path(POSTED_TRACKER_FILE)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("urls", []))
    return set()


def save_posted_urls(urls: set[str]) -> None:
    """
    Persist the updated set of processed URLs back to disk.

    Called ONCE at the very end of each pipeline run so we're not writing
    to disk after every single article.
    """
    Path(POSTED_TRACKER_FILE).write_text(
        json.dumps(
            {
                "urls":    sorted(urls),   # sorted → consistent diffs in git
                "updated": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"Saved {len(urls)} URL(s) to {POSTED_TRACKER_FILE}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — NEWS FETCHING  (NewsAPI)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_articles() -> list[dict]:
    """
    Query NewsAPI for recent English articles matching NEWS_QUERY.

    Returns a list of article dicts, each containing at minimum:
      - title, description, content (truncated on free tier), url,
        publishedAt, source.name

    Free-tier note: the `content` field is cut off at ~200 characters.
    Paid tiers return the full article body.
    If the request fails for any reason, returns an empty list so the
    rest of the pipeline can exit gracefully.
    """
    log.info(f'Fetching articles — query: "{NEWS_QUERY}"')

    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        NEWS_QUERY,
                "language": "en",          # English articles only
                "sortBy":   "publishedAt", # Most recent first
                "pageSize": NEWS_PAGE_SIZE,
                "apiKey":   NEWSAPI_KEY,
            },
            timeout=20,   # Give up after 20 s to avoid hanging Railway workers
        )
        response.raise_for_status()   # Raises an exception for 4xx / 5xx

        articles = response.json().get("articles", [])
        log.info(f"NewsAPI returned {len(articles)} article(s)")
        return articles

    except requests.exceptions.HTTPError as exc:
        log.error(f"NewsAPI HTTP error {exc.response.status_code}: {exc.response.text}")
        return []
    except requests.exceptions.RequestException as exc:
        log.error(f"NewsAPI request failed: {exc}")
        return []


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — AI ANALYSIS  (Claude)
# ═════════════════════════════════════════════════════════════════════════════

class ArticleAnalysis(BaseModel):
    """
    The exact shape of data we want Claude to return for every article.

    We pass this Pydantic class to client.messages.parse() which:
      1. Converts it into a JSON Schema and sends it to the Claude API
      2. Guarantees Claude's reply matches that schema
      3. Returns a validated Python object (no manual json.loads() needed)

    Field-by-field:
      relevance_score     — 1–10 integer judging audience fit
      rewritten_title     — SEO-friendly headline (max ~70 chars)
      rewritten_content   — Full HTML article body (300–600 words)
      category            — One of our six predefined categories
      tags                — 3–5 lowercase, hyphenated SEO tags
    """
    relevance_score:   int
    rewritten_title:   str
    rewritten_content: str
    category:          Literal[
                           "News", "Automotive", "Industrial",
                           "DIY", "Technology", "Materials"
                       ]
    tags:              List[str]   # e.g. ["3d-printing", "spare-parts"]


def analyze_with_claude(
    client: Anthropic,
    article: dict,
) -> Optional[ArticleAnalysis]:
    """
    Send one article to Claude and receive a structured, validated response.

    Claude is asked to:
      ① Score the article's relevance for a 3D-printing spare-parts audience
      ② Rewrite the article in a professional, engaging HTML format
      ③ Assign the best-fit category from our predefined list
      ④ Generate 3–5 lowercase SEO tags

    Returns an ArticleAnalysis object on success, or None on any error.
    The caller decides whether the score is high enough to publish.
    """
    # ── Pull fields from the NewsAPI article dict ─────────────────────────
    title       = article.get("title")       or "Untitled"
    description = article.get("description") or "(no description)"
    content     = article.get("content")     or "(no content preview)"
    source_name = article.get("source", {}).get("name", "Unknown source")
    published   = article.get("publishedAt", "")
    url         = article.get("url", "")

    # ── Build the prompt Claude will receive ──────────────────────────────
    # We explain the site's focus, provide the raw article data, and give
    # clear instructions for each output field.
    prompt = f"""You are a senior content editor for 3dprintedspareparts.com — a professional
blog serving engineers, hobbyists, and manufacturers interested in 3D-printed
spare parts, replacement components, and additive manufacturing applications.

Analyse the news article below and return a structured response.

━━━ ORIGINAL ARTICLE ━━━
Title:       {title}
Source:      {source_name}
Published:   {published}
URL:         {url}
Description: {description}
Content:     {content}
━━━━━━━━━━━━━━━━━━━━━━━━

SCORING GUIDE  (relevance_score — integer 1 to 10):
• 8–10  → Directly about 3D-printed spare/replacement parts, automotive or
          industrial 3D printing, new materials for part manufacturing,
          supply chain disruption solved by 3D printing
• 5–7   → Related to additive manufacturing, 3D printing technology broadly,
          repair culture, or manufacturing innovation
• 1–4   → Only tangentially connected; unlikely to interest our readers

REWRITING RULES:
• rewritten_title    — concise, keyword-rich, max 70 characters
• rewritten_content  — 350–550 words, HTML-formatted with <p> and <h2> tags,
                       professional yet approachable tone for a technical
                       audience; end with an attribution paragraph:
                       <p><em>Originally reported by {source_name}.</em></p>

CATEGORY — pick exactly one:
  News | Automotive | Industrial | DIY | Technology | Materials

TAGS — 3 to 5 items, all lowercase, hyphenated where multi-word:
  e.g. "3d-printing", "spare-parts", "automotive-repair", "fdm"
"""

    try:
        log.info(f'  → Sending to Claude: "{title[:65]}…"')

        # ── The key Claude API call ───────────────────────────────────────
        # client.messages.parse() is the recommended way to get structured
        # output.  It:
        #   • Converts ArticleAnalysis into a JSON Schema automatically
        #   • Sends that schema to the API via output_config
        #   • Validates Claude's JSON response against the Pydantic model
        #   • Returns response.parsed_output as a real Python object
        #
        # If you need longer output or are hitting timeouts, switch to the
        # streaming approach:  client.messages.stream(...).get_final_message()
        response = client.messages.parse(
            model=CLAUDE_MODEL,
            max_tokens=2048,   # Enough for ~600 words of HTML + JSON overhead
            messages=[{"role": "user", "content": prompt}],
            output_format=ArticleAnalysis,   # ← Structured output magic ✨
        )

        result = response.parsed_output

        log.info(
            f"  ✓ Score {result.relevance_score}/10 | "
            f"Category: {result.category} | "
            f"Tags: {', '.join(result.tags)}"
        )
        return result

    except Exception as exc:
        # Catching broadly so a single bad article doesn't crash the pipeline.
        # Common causes: network blip, malformed article text, schema mismatch.
        log.error(f"  ✗ Claude error: {exc}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — WORDPRESS PUBLISHING  (WordPress.com REST API)
# ═════════════════════════════════════════════════════════════════════════════

def publish_to_wordpress(analysis: ArticleAnalysis, source_url: str) -> bool:
    """
    Create a new published post on WordPress.com using their REST API v1.1.

    Endpoint:
      POST https://public-api.wordpress.com/rest/v1.1/sites/{SITE_ID}/posts/new

    Authentication:
      Bearer token passed in the Authorization header.
      Get yours at: https://developer.wordpress.com/apps/
      (Create an app → Authorize → copy the access token into WP_TOKEN)

    The WordPress.com API v1.1 accepts:
      - categories  as a plain string  (e.g. "Automotive")
      - tags        as a comma-separated string

    Returns True if the post was created successfully, False otherwise.
    """
    endpoint = (
        f"https://public-api.wordpress.com/rest/v1.1"
        f"/sites/{WP_SITE_ID}/posts/new"
    )

    payload = {
        "title":      analysis.rewritten_title,
        "content":    analysis.rewritten_content,
        "status":     "publish",                     # Publish immediately
        "categories": analysis.category,             # Category name (string)
        "tags":       ", ".join(analysis.tags),      # Comma-separated tag names
        "format":     "standard",
    }

    try:
        log.info(f'  → Publishing: "{analysis.rewritten_title}"')

        resp = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {WP_TOKEN}"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()

        post = resp.json()
        log.info(
            f"  ✅ Published!  ID={post.get('ID')}  "
            f"URL={post.get('URL', 'unknown')}"
        )
        return True

    except requests.exceptions.HTTPError as exc:
        # Log the full API error body — often contains a helpful message
        # like "invalid_token" or "unauthorized".
        try:
            detail = exc.response.json()
        except Exception:
            detail = exc.response.text
        log.error(f"  ✗ WordPress HTTP {exc.response.status_code}: {detail}")
        return False

    except requests.exceptions.RequestException as exc:
        log.error(f"  ✗ WordPress request error: {exc}")
        return False


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CONFIG VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_env() -> bool:
    """
    Check that all four required environment variables have been set.

    Called at the very start of run() so the pipeline fails fast with a
    clear message rather than crashing mid-way with a cryptic error.
    """
    required = {
        "NEWSAPI_KEY":        NEWSAPI_KEY,
        "ANTHROPIC_API_KEY":  ANTHROPIC_KEY,
        "WP_SITE_ID":         WP_SITE_ID,
        "WP_TOKEN":           WP_TOKEN,
    }
    missing = [key for key, val in required.items() if not val]

    if missing:
        log.error("❌ Missing environment variables: " + ", ".join(missing))
        log.error("   Copy .env.example → .env and fill in the missing values.")
        return False

    return True


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN PIPELINE  (orchestrates all steps)
# ═════════════════════════════════════════════════════════════════════════════

def run() -> None:
    """
    The main pipeline function — called once per scheduled run.

    Full flow:
      1.  Validate that all env vars are present
      2.  Load the list of already-processed article URLs
      3.  Fetch fresh articles from NewsAPI
      4.  For each article:
          a. Skip if the URL was already processed (no duplicates)
          b. Send to Claude for analysis
          c. Skip if relevance score < RELEVANCE_THRESHOLD (default: 7)
          d. Publish to WordPress.com
          e. Record the URL so it won't be processed again
      5.  Save the updated URL list to disk
      6.  Print a summary of what happened this run
    """
    log.info("=" * 60)
    log.info("  3D Printed Spare Parts — Content Pipeline")
    log.info(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info("=" * 60)

    # ── Step 1: Validate config ───────────────────────────────────────────
    if not validate_env():
        log.error("Pipeline aborted — fix the missing env vars above.")
        return

    # ── Initialise Claude client ──────────────────────────────────────────
    # Anthropic() reads ANTHROPIC_API_KEY from the environment automatically.
    claude = Anthropic()

    # ── Step 2: Load duplicate tracker ───────────────────────────────────
    posted_urls = load_posted_urls()
    log.info(f"Duplicate tracker: {len(posted_urls)} URL(s) already on record")

    # ── Step 3: Fetch articles ────────────────────────────────────────────
    articles = fetch_articles()
    if not articles:
        log.warning("No articles returned by NewsAPI — nothing to do, exiting.")
        return

    # ── Per-run counters (for the summary at the end) ─────────────────────
    total         = len(articles)
    skipped_dup   = 0   # Already in our tracker
    skipped_score = 0   # Scored below the threshold
    published     = 0   # Successfully posted to WordPress
    errors        = 0   # Any kind of failure

    # ── Step 4: Process each article ─────────────────────────────────────
    for idx, article in enumerate(articles, start=1):
        url   = article.get("url", "")
        title = article.get("title", "No title")

        log.info(f"\n[{idx}/{total}]  {title[:80]}")

        # 4a — Every article needs a URL (we use it as a unique identifier)
        if not url:
            log.warning("  No URL found — cannot track this article, skipping.")
            errors += 1
            continue

        # 4b — Skip if we've already processed this URL
        if url in posted_urls:
            log.info("  Already processed — skipping.")
            skipped_dup += 1
            continue

        # 4c — Ask Claude to analyse the article
        analysis = analyze_with_claude(claude, article)

        if analysis is None:
            log.error("  Analysis failed — article will be retried next run.")
            # Do NOT add to posted_urls so we retry on the next scheduled run.
            errors += 1
            continue

        # 4d — Only publish if the score meets our quality bar
        if analysis.relevance_score < RELEVANCE_THRESHOLD:
            log.info(
                f"  Score {analysis.relevance_score}/10 is below the "
                f"{RELEVANCE_THRESHOLD}/10 threshold — not publishing."
            )
            skipped_score += 1
            # Mark as seen so we don't waste Claude credits re-analysing it.
            posted_urls.add(url)
            continue

        # 4e — Score is good — publish!
        log.info(
            f"  Score {analysis.relevance_score}/10 ✓ — "
            "above threshold, publishing…"
        )
        success = publish_to_wordpress(analysis, url)

        # Always record the URL whether publishing succeeded or not.
        # If WordPress fails once, we won't re-analyse next run — which is
        # intentional to avoid flooding WP with duplicate drafts.
        posted_urls.add(url)

        if success:
            published += 1
        else:
            errors += 1

        # Small pause between Claude calls — avoids hitting rate limits and
        # is generally polite to the APIs we're using.
        time.sleep(DELAY_BETWEEN_CALLS)

    # ── Step 5: Save updated tracker ─────────────────────────────────────
    save_posted_urls(posted_urls)

    # ── Step 6: Print summary ─────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("  Run Summary")
    log.info(f"  Articles fetched:        {total}")
    log.info(f"  Skipped (seen before):   {skipped_dup}")
    log.info(f"  Skipped (score too low): {skipped_score}")
    log.info(f"  Published to WordPress:  {published}")
    log.info(f"  Errors:                  {errors}")
    log.info("=" * 60)
    log.info("  Pipeline finished.")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run()
