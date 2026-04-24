"""
Data Generator — Sends synthetic production-style traffic to the integrated
Paperless-ngx + ml_hooks stack.

Simulates realistic user behavior against the real Paperless REST API:
  1. Uploads scanned documents           → POST /api/documents/post_document/
  2. Executes search queries              → GET  /api/search/?query=...
  3. Submits HTR corrections              → POST /api/ml/feedback/  (kind=htr_correction)
  4. Provides search-result feedback      → POST /api/ml/feedback/  (kind=search_click|search_rating)

The ml_hooks Feedback model unifies what used to be separate corrections and
search-feedback endpoints in the stub API; this generator posts against the
unified endpoint, using `kind` as the discriminator.

All requests carry a Paperless API token via the `Authorization: Token <key>`
header. The generator reads the token from --paperless-token, $PAPERLESS_TOKEN,
or prompts for it.

Usage:
  python generator.py \
      --paperless-url   http://localhost:8000 \
      --paperless-token $PAPERLESS_TOKEN \
      --rate 2.0 --duration 300
"""

import argparse
import io
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime

import httpx
import numpy as np
from PIL import Image, ImageDraw
from faker import Faker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

fake = Faker()
Faker.seed(42)
random.seed(42)

# ── Synthetic data generators ──────────────────

# Realistic search queries for a document management system
SEARCH_QUERIES = [
    "budget report 2024", "meeting minutes march", "invoice",
    "employee handbook update", "grant proposal", "equipment purchase",
    "travel reimbursement", "committee meeting", "course syllabus",
    "research paper", "lab safety protocol", "student enrollment",
    "financial audit", "facilities maintenance", "IT support",
    "curriculum committee", "parking permit", "building access",
    "conference registration", "library acquisition", "departmental budget",
    "hiring committee", "tenure review", "annual performance review",
    "room reservation", "supply order", "safety inspection",
    "accreditation self-study", "strategic plan", "alumni newsletter",
]

# Handwriting-like correction texts
CORRECTION_TEXTS = [
    "Please review and approve by Friday",
    "Budget approved - see attached memo",
    "Meeting rescheduled to 3:00 PM",
    "Forward to department chair",
    "Revised per committee feedback",
    "Signature required on page 3",
    "See handwritten notes in margin",
    "Updated figures for Q3",
    "Action items from today's meeting",
    "Draft - do not distribute",
    "Confidential - internal use only",
    "Approved with minor revisions",
    "Return to sender - wrong department",
    "File under grants and contracts",
    "Urgent - deadline approaching",
]


def generate_synthetic_page() -> bytes:
    """Create a synthetic document page image with text."""
    width, height = 850, 1100  # ~letter size at 100 DPI
    img = Image.new("RGB", (width, height), color=(252, 251, 248))
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((50, 40), fake.company(), fill=(30, 30, 30))
    draw.text((50, 70), fake.address().replace("\n", ", "), fill=(80, 80, 80))
    draw.line([(50, 100), (800, 100)], fill=(150, 150, 150), width=2)

    # Body text
    y = 130
    for _ in range(random.randint(8, 20)):
        line = fake.sentence(nb_words=random.randint(6, 14))
        draw.text((50, y), line, fill=(40, 40, 40))
        y += 22
        if y > 900:
            break

    # Handwritten scribbles (dark ink — should trigger region detection)
    for _ in range(random.randint(1, 3)):
        x_start = random.randint(50, 400)
        y_start = random.randint(200, 800)
        points = [
            (x_start + i * 8, y_start + random.randint(-3, 3))
            for i in range(random.randint(10, 30))
        ]
        draw.line(points, fill=(20, 20, 150), width=3)

    # Margin note in red
    if random.random() < 0.4:
        note_y = random.randint(200, 700)
        draw.text((650, note_y), random.choice(CORRECTION_TEXTS)[:20], fill=(180, 30, 30))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Paperless traffic generator ────────────────

class PaperlessTrafficGenerator:
    """
    Posts synthetic traffic against the real Paperless REST API.

    State tracked across the run (for realistic follow-up traffic):
      uploaded_doc_ids   list[int]           — integer IDs returned by Paperless upload
      last_search_ids    dict[str, list[int]] — query -> list of doc ids recently returned
    """

    def __init__(self, paperless_url: str, paperless_token: str):
        self.paperless_url = paperless_url.rstrip("/")
        self.paperless_token = paperless_token
        self.client = httpx.Client(
            timeout=30.0,
            headers={"Authorization": f"Token {paperless_token}"},
        )
        self.uploaded_doc_ids: list[int] = []
        self.last_search_ids: dict[str, list[int]] = {}
        self.stats = {
            "uploads": 0, "corrections": 0, "searches": 0, "feedback": 0, "errors": 0,
        }

    # ── Individual actions ─────────────────────

    def do_upload(self):
        """Upload a synthetic page image to Paperless."""
        img_bytes = generate_synthetic_page()
        filename = f"{fake.file_name(extension='png')}"
        title = f"synthetic-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{random.randint(1000, 9999)}"

        try:
            resp = self.client.post(
                f"{self.paperless_url}/api/documents/post_document/",
                files={"document": (filename, img_bytes, "image/png")},
                data={"title": title},
            )
            resp.raise_for_status()
            # Paperless returns a task UUID on async ingest, not the doc ID.
            # We don't wait for ingest here; we rely on the global search
            # endpoint to discover IDs for feedback traffic below.
            self.stats["uploads"] += 1
            log.info(f"[UPLOAD] file={filename} title={title}")
        except httpx.HTTPStatusError as e:
            self.stats["errors"] += 1
            log.error(f"[UPLOAD ERROR] HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            self.stats["errors"] += 1
            log.error(f"[UPLOAD ERROR] {e}")

    def do_search(self):
        """Execute a global search query. Remember the returned doc IDs for feedback."""
        query = random.choice(SEARCH_QUERIES)
        session_id = str(uuid.uuid4())

        try:
            resp = self.client.get(
                f"{self.paperless_url}/api/search/",
                params={"query": query, "session_id": session_id},
            )
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("documents", []) or []
            doc_ids = [d["id"] for d in docs if isinstance(d, dict) and "id" in d]
            if doc_ids:
                self.last_search_ids[query] = doc_ids
                # Also populate uploaded_doc_ids so corrections can target real docs,
                # in case the generator starts with an empty stack.
                for did in doc_ids:
                    if did not in self.uploaded_doc_ids:
                        self.uploaded_doc_ids.append(did)
            self.stats["searches"] += 1
            log.info(
                f"[SEARCH] '{query}' → {len(doc_ids)} results"
                + (f" (semantic_added={data.get('ml_semantic_added', 0)})"
                   if 'ml_semantic_added' in data else "")
            )
        except httpx.HTTPStatusError as e:
            self.stats["errors"] += 1
            log.error(f"[SEARCH ERROR] HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            self.stats["errors"] += 1
            log.error(f"[SEARCH ERROR] {e}")

    def do_correction(self):
        """Post an HTR correction for a random uploaded document."""
        if not self.uploaded_doc_ids:
            return
        doc_id = random.choice(self.uploaded_doc_ids)
        text = random.choice(CORRECTION_TEXTS)
        try:
            resp = self.client.post(
                f"{self.paperless_url}/api/ml/feedback/",
                json={
                    "document": doc_id,
                    "kind": "htr_correction",
                    "correction_text": text,
                },
            )
            resp.raise_for_status()
            self.stats["corrections"] += 1
            log.info(f"[CORRECTION] doc={doc_id} text='{text[:40]}...'")
        except httpx.HTTPStatusError as e:
            self.stats["errors"] += 1
            log.error(f"[CORRECTION ERROR] HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            self.stats["errors"] += 1
            log.error(f"[CORRECTION ERROR] {e}")

    def do_feedback(self):
        """
        Post feedback on a recent search result. 60% clicks, 25% thumbs_up, 15% thumbs_down.
        Corresponds to ml_hooks Feedback.Kind.SEARCH_CLICK / SEARCH_RATING.
        """
        if not self.last_search_ids:
            return
        query, doc_ids = random.choice(list(self.last_search_ids.items()))
        if not doc_ids:
            return
        doc_id = random.choice(doc_ids)

        roll = random.random()
        if roll < 0.60:
            kind = "search_click"
            body = {"document": doc_id, "kind": kind, "query_text": query}
            label = "click"
        elif roll < 0.85:
            kind = "search_rating"
            body = {"document": doc_id, "kind": kind, "rating": 1, "query_text": query}
            label = "thumbs_up"
        else:
            kind = "search_rating"
            body = {"document": doc_id, "kind": kind, "rating": 0, "query_text": query}
            label = "thumbs_down"

        try:
            resp = self.client.post(
                f"{self.paperless_url}/api/ml/feedback/",
                json=body,
            )
            resp.raise_for_status()
            self.stats["feedback"] += 1
            log.info(f"[FEEDBACK] {label} on doc={doc_id} query='{query[:30]}'")
        except httpx.HTTPStatusError as e:
            self.stats["errors"] += 1
            log.error(f"[FEEDBACK ERROR] HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            self.stats["errors"] += 1
            log.error(f"[FEEDBACK ERROR] {e}")

    # ── Main loop ──────────────────────────────

    def run(self, rate: float, duration: int, upload_only: bool = False):
        """
        rate: average events per second
        duration: total runtime in seconds
        upload_only: if True, only generates document uploads (no searches,
          corrections, or feedback). Use this when running alongside
          behavior_emulator which owns the corrections + feedback streams.

        Traffic pattern (normal mode):
          First 20%: upload-heavy (build up documents before searching them)
          Middle 60%: mixed traffic
          Last 20%: search + feedback heavy (realistic ongoing usage)
        """
        log.info(
            "Starting traffic generator: rate=%s/s, duration=%ss, upload_only=%s",
            rate, duration, upload_only,
        )
        log.info(f"Target Paperless: {self.paperless_url}")

        start = time.time()
        elapsed = 0.0
        while elapsed < duration:
            if upload_only:
                action = self.do_upload
            else:
                progress = elapsed / duration
                if progress < 0.2:
                    weights = [0.70, 0.15, 0.10, 0.05]  # upload, search, correction, feedback
                elif progress < 0.8:
                    weights = [0.25, 0.35, 0.20, 0.20]
                else:
                    weights = [0.10, 0.40, 0.15, 0.35]
                action = random.choices(
                    [self.do_upload, self.do_search, self.do_correction, self.do_feedback],
                    weights=weights,
                )[0]
            action()

            sleep_time = max(0.1, (1.0 / rate) + random.uniform(-0.2, 0.2))
            time.sleep(sleep_time)
            elapsed = time.time() - start

        log.info(f"Generator finished. Stats: {json.dumps(self.stats)}")
        log.info(
            f"  Uploaded {self.stats['uploads']} docs, "
            f"{self.stats['searches']} searches, "
            f"{self.stats['corrections']} corrections, "
            f"{self.stats['feedback']} feedback events, "
            f"{self.stats['errors']} errors"
        )


# ── CLI ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic production-style traffic generator for the integrated "
                    "Paperless-ngx + ml_hooks stack.",
    )
    parser.add_argument(
        "--paperless-url",
        default=os.environ.get("PAPERLESS_URL", "http://localhost:8000"),
        help="Paperless base URL (default: $PAPERLESS_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--paperless-token",
        default=os.environ.get("PAPERLESS_TOKEN", ""),
        help="Paperless API token (default: $PAPERLESS_TOKEN). Required.",
    )
    parser.add_argument("--rate", type=float, default=2.0,
                        help="Average events per second (default: 2.0)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Total runtime in seconds (default: 300)")
    parser.add_argument("--upload-only", action="store_true",
                        default=os.environ.get("UPLOAD_ONLY", "").lower() in ("1", "true", "yes"),
                        help="Only generate document uploads — no searches, corrections, "
                             "or feedback. Intended for running alongside behavior_emulator "
                             "which owns the corrections + feedback streams.")
    args = parser.parse_args()

    if not args.paperless_token:
        parser.error(
            "Paperless API token is required. Pass --paperless-token "
            "or set $PAPERLESS_TOKEN."
        )

    gen = PaperlessTrafficGenerator(args.paperless_url, args.paperless_token)
    gen.run(args.rate, args.duration, upload_only=args.upload_only)


if __name__ == "__main__":
    main()
