#!/usr/bin/env python3
"""
3D Printed Spare Parts — Automated Content Pipeline
====================================================

HOW IT WORKS (high-level):
  1. NewsAPI       → Fetch recent English news about "3D printed spare parts"
  2. Claude AI     → Score relevance, rewrite in professional tone, assign
                     a category, and generate SEO tags  (structured output)
  3. Gemini Imagen → Generate a featured image for every approved article:
                     a. Claude writes an optimised Imagen prompt
                     b. Gemini Imagen renders a photorealistic 16:9 image
                     c. The image is uploaded to the WordPress.com media library
  4. WordPress.com → Publish any article scoring 7+ with its featured image set

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
import requests                          # HTTP calls to NewsAPI and WordPress.com
from anthropic import Anthropic          # Official Claude SDK
from dotenv import load_dotenv           # Reads .env into os.environ at startup
from google import genai as google_genai # Official Google Generative AI SDK
from google.genai import types as genai_types
from pydantic import BaseModel           # Data-validation for Claude's structured output


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
GEMINI_KEY    = os.getenv("GEMINI_API_KEY")       # Google AI Studio key for Imagen

# ── Pipeline settings — feel free to tweak these ─────────────────────────────
NEWS_QUERY          = "3D printed spare parts"
NEWS_PAGE_SIZE      = 10     # How many articles to pull per run (max 100 on free NewsAPI)
RELEVANCE_THRESHOLD = 7      # Minimum score out of 10 needed to publish
CLAUDE_MODEL        = "claude-opus-4-6"
GEMINI_IMAGE_MODEL  = "imagen-3.0-generate-002"  # Imagen 3 — highest quality
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
# SECTION 6 — IMAGE PROMPT GENERATION  (Claude)
# ═════════════════════════════════════════════════════════════════════════════

def generate_image_prompt(
    client: Anthropic,
    analysis: ArticleAnalysis,
) -> Optional[str]:
    """
    Ask Claude to write an optimised text-to-image prompt for Gemini Imagen.

    Why Claude for this step?
      • Claude has read the full article and understands the nuance
      • A well-crafted prompt produces significantly better Imagen output
      • We keep the prompt industrial/technical to match our site's audience

    Prompt rules enforced:
      - Photorealistic style, no text or overlays in the image
      - 3D printing / manufacturing subject matter
      - ≤ 200 characters (Imagen works best with concise, vivid prompts)
      - Landscape 16:9 framing (matches WordPress featured-image dimensions)

    Returns the prompt string on success, or None if the Claude call fails.
    """
    prompt = f"""You are an expert at writing prompts for AI image generation (Google Imagen).

Write a single image generation prompt for the blog post below. The image will
be used as the featured hero image on a 3D printing & spare parts blog.

Article title: {analysis.rewritten_title}
Category:      {analysis.category}
Tags:          {', '.join(analysis.tags)}

STRICT RULES — your prompt MUST follow all of these:
1. Photorealistic style — cinematic lighting, high detail
2. Subject must relate clearly to 3D printing, manufacturing, or spare parts
3. No people, no faces, no text, no logos in the image
4. Landscape / wide-angle composition (16:9 format)
5. Maximum 200 characters in total
6. Return ONLY the prompt text — no explanations, no quotes, no labels

Example of a good prompt:
  Close-up macro photo of a white FDM 3D printer nozzle depositing molten
  plastic filament layer by layer, industrial workshop background, shallow DOF
"""

    try:
        log.info("  → Generating image prompt with Claude…")

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        image_prompt = response.content[0].text.strip()

        # Truncate hard at 400 chars as a safety net — Imagen can handle long
        # prompts but shorter ones are more predictable.
        if len(image_prompt) > 400:
            image_prompt = image_prompt[:400]

        log.info(f'  ✓ Image prompt: "{image_prompt[:100]}{"…" if len(image_prompt) > 100 else ""}"')
        return image_prompt

    except Exception as exc:
        log.error(f"  ✗ Image prompt generation failed: {exc}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — IMAGE GENERATION  (Google Gemini Imagen)
# ═════════════════════════════════════════════════════════════════════════════

def generate_image_with_gemini(
    gemini_client: google_genai.Client,
    prompt: str,
) -> Optional[bytes]:
    """
    Call the Gemini Imagen API to generate a single photorealistic image.

    Model used: imagen-3.0-generate-002  (Imagen 3 — highest quality tier)
    Output:     PNG bytes at 1024×576 (16:9) — ready for direct upload

    Why Imagen 3?
      • Best-in-class photorealism for product/industrial photography style
      • Consistent, clean results for technical subject matter

    Returns raw PNG bytes on success, or None on any failure.
    Failures here are non-fatal — the post will still be published, just
    without a featured image.
    """
    try:
        log.info(f"  → Sending prompt to Gemini Imagen ({GEMINI_IMAGE_MODEL})…")

        response = gemini_client.models.generate_images(
            model=GEMINI_IMAGE_MODEL,
            prompt=prompt,
            config=genai_types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9",           # Wide landscape — ideal for blog heroes
                safety_filter_level="block_only_high",  # Allow industrial/mechanical content
                person_generation="dont_allow",          # No people — avoids edge-case issues
            ),
        )

        if not response.generated_images:
            log.warning("  ⚠ Gemini returned no images (prompt may have been filtered).")
            return None

        image_bytes = response.generated_images[0].image.image_bytes
        log.info(f"  ✓ Image generated ({len(image_bytes):,} bytes)")
        return image_bytes

    except Exception as exc:
        log.error(f"  ✗ Gemini Imagen error: {exc}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — WORDPRESS MEDIA UPLOAD
# ═════════════════════════════════════════════════════════════════════════════

def upload_image_to_wordpress(
    image_bytes: bytes,
    filename: str,
) -> Optional[int]:
    """
    Upload a PNG image to the WordPress.com media library.

    Endpoint:
      POST https://public-api.wordpress.com/rest/v1.1/sites/{SITE_ID}/media/new

    The image is sent as multipart/form-data.  WordPress.com returns the
    assigned media ID which we then pass to publish_to_wordpress() so the
    post's featured image is set automatically.

    Returns the integer media ID on success, or None on any failure.
    """
    endpoint = (
        f"https://public-api.wordpress.com/rest/v1.1"
        f"/sites/{WP_SITE_ID}/media/new"
    )

    try:
        log.info(f"  → Uploading image to WordPress media library ({filename})…")

        resp = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {WP_TOKEN}"},
            files={"media[]": (filename, image_bytes, "image/png")},
            timeout=60,   # Image upload can be slow on Railway — give it time
        )
        resp.raise_for_status()

        data = resp.json()
        # The API returns a 'media' list even for single uploads
        media_list = data.get("media", [])
        if not media_list:
            log.error(f"  ✗ WordPress media upload returned empty media list: {data}")
            return None

        media_id = media_list[0].get("ID")
        media_url = media_list[0].get("URL", "unknown")
        log.info(f"  ✓ Media uploaded  ID={media_id}  URL={media_url}")
        return int(media_id) if media_id else None

    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json()
        except Exception:
            detail = exc.response.text
        log.error(f"  ✗ WordPress media upload HTTP {exc.response.status_code}: {detail}")
        return None

    except requests.exceptions.RequestException as exc:
        log.error(f"  ✗ WordPress media upload request error: {exc}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — WORDPRESS PUBLISHING  (WordPress.com REST API)
# ═════════════════════════════════════════════════════════════════════════════

def publish_to_wordpress(
    analysis: ArticleAnalysis,
    source_url: str,
    featured_image_id: Optional[int] = None,
) -> bool:
    """
    Create a new published post on WordPress.com using their REST API v1.1.

    Endpoint:
      POST https://public-api.wordpress.com/rest/v1.1/sites/{SITE_ID}/posts/new

    Authentication:
      Bearer token passed in the Authorization header.
      Get yours at: https://developer.wordpress.com/apps/
      (Create an app → Authorize → copy the access token into WP_TOKEN)

    The WordPress.com API v1.1 accepts:
      - categories       as a plain string  (e.g. "Automotive")
      - tags             as a comma-separated string
      - featured_image   as a media ID integer (optional; omitted if None)

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

    # Only attach the featured image if we successfully generated and uploaded one
    if featured_image_id is not None:
        payload["featured_image"] = str(featured_image_id)
        log.info(f"  → Attaching featured image  ID={featured_image_id}")

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
        image_note = f"  (featured image: {featured_image_id})" if featured_image_id else ""
        log.info(
            f"  ✅ Published!  ID={post.get('ID')}  "
            f"URL={post.get('URL', 'unknown')}{image_note}"
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
# SECTION 10 — CONFIG VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_env() -> bool:
    """
    Check that all five required environment variables have been set.

    Called at the very start of run() so the pipeline fails fast with a
    clear message rather than crashing mid-way with a cryptic error.
    """
    required = {
        "NEWSAPI_KEY":        NEWSAPI_KEY,
        "ANTHROPIC_API_KEY":  ANTHROPIC_KEY,
        "WP_SITE_ID":         WP_SITE_ID,
        "WP_TOKEN":           WP_TOKEN,
        "GEMINI_API_KEY":     GEMINI_KEY,
    }
    missing = [key for key, val in required.items() if not val]

    if missing:
        log.error("❌ Missing environment variables: " + ", ".join(missing))
        log.error("   Copy .env.example → .env and fill in the missing values.")
        return False

    return True


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — MAIN PIPELINE  (orchestrates all steps)
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
          d. Ask Claude to generate an Imagen prompt for the article
          e. Send prompt to Gemini Imagen → receive PNG bytes
          f. Upload the image to WordPress.com media library
          g. Publish post to WordPress.com with featured image attached
          h. Record the URL so it won't be processed again
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

    # ── Initialise API clients ────────────────────────────────────────────
    # Anthropic() reads ANTHROPIC_API_KEY from the environment automatically.
    claude = Anthropic()
    # google_genai.Client() uses the provided api_key.
    gemini = google_genai.Client(api_key=GEMINI_KEY)

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
    images_added  = 0   # Posts that got a featured image successfully

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

        # 4e — Score is good — generate a featured image ──────────────────
        log.info(
            f"  Score {analysis.relevance_score}/10 ✓ — "
            "above threshold, generating featured image…"
        )

        featured_image_id: Optional[int] = None   # Will stay None if any step fails

        # Step 1: Claude writes the Imagen prompt
        image_prompt = generate_image_prompt(claude, analysis)

        if image_prompt:
            # Step 2: Gemini Imagen renders the image
            image_bytes = generate_image_with_gemini(gemini, image_prompt)

            if image_bytes:
                # Step 3: Upload the PNG to the WordPress media library
                safe_slug = (
                    analysis.rewritten_title.lower()
                    .replace(" ", "-")
                    [:40]                         # Limit filename length
                    .rstrip("-")
                )
                filename = f"{safe_slug}-{int(time.time())}.png"
                featured_image_id = upload_image_to_wordpress(image_bytes, filename)

                if featured_image_id:
                    images_added += 1
                else:
                    log.warning("  ⚠ Image upload failed — publishing without featured image.")
            else:
                log.warning("  ⚠ Imagen returned no image — publishing without featured image.")
        else:
            log.warning("  ⚠ Image prompt generation failed — publishing without featured image.")

        # 4f — Publish to WordPress (with or without featured image)
        log.info("  → Publishing to WordPress…")
        success = publish_to_wordpress(analysis, url, featured_image_id=featured_image_id)

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
    log.info(f"  With featured image:     {images_added}")
    log.info(f"  Errors:                  {errors}")
    log.info("=" * 60)
    log.info("  Pipeline finished.")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run()
