

"""
build_guideline_corpus.py - Version 3

Builds a medical guideline corpus from manually verified seed URLs.

Output schema:
{
    "id",
    "text",
    "disease_tag",
    "source_url",
    "source_date"
}

Important:
- source_date is extracted from the source page.
- Today's date is NEVER used as a fake fallback.
- If no trustworthy date can be found, source_date is None.
"""

import time
import json
import hashlib
import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import parser as dparser
import pandas as pd
import urllib.robotparser as robotparser


# ============================================================
# Configuration
# ============================================================

HEADERS = {
    "User-Agent": (
        "academic-research-bot/1.0 "
        "(medical-information-retrieval-research)"
    )
}

MIN_CHUNK_WORDS = 40
MAX_CHUNK_WORDS = 180


# ============================================================
# Robots
# ============================================================

def robots_allowed(url):

    from urllib.parse import urlparse

    parsed = urlparse(url)

    rp = robotparser.RobotFileParser()

    rp.set_url(
        f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    )

    try:

        rp.read()

        return rp.can_fetch(
            HEADERS["User-Agent"],
            url
        )

    except Exception:

        # Fail closed
        return False


# ============================================================
# Date helpers
# ============================================================

def normalize_date(value):
    """
    Convert a date-like value into YYYY-MM-DD.

    Returns None if the value cannot be parsed reliably.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    # Ignore obvious non-date values
    if value.lower() in {
        "unknown",
        "n/a",
        "na",
        "none",
        "null",
    }:
        return None

    try:

        parsed = dparser.parse(
            value,
            fuzzy=True
        )

        # Reject implausible dates
        if parsed.year < 1990:
            return None

        if parsed.year > datetime.now().year + 1:
            return None

        return str(parsed.date())

    except Exception:

        return None


def extract_real_review_date(soup):
    """
    Extract the most trustworthy publication/review/update date
    available on the page.

    Priority:
        1. JSON-LD structured metadata
        2. OpenGraph / article metadata
        3. <time datetime="">
        4. Visible text patterns

    Returns:
        YYYY-MM-DD string or None
    """

    # --------------------------------------------------------
    # 1. JSON-LD structured data
    # --------------------------------------------------------

    jsonld_dates = []

    for script in soup.find_all(
        "script",
        type="application/ld+json"
    ):

        try:

            data = json.loads(
                script.string or script.get_text()
            )

        except Exception:

            continue

        objects = []

        if isinstance(data, dict):

            objects.append(data)

            # Handle @graph
            if isinstance(data.get("@graph"), list):
                objects.extend(data["@graph"])

        elif isinstance(data, list):

            objects.extend(data)

        for obj in objects:

            if not isinstance(obj, dict):
                continue

            # Prefer modified/review dates over publication date
            for key in [
                "dateModified",
                "dateReviewed",
                "datePublished",
            ]:

                if key in obj:

                    parsed = normalize_date(
                        obj.get(key)
                    )

                    if parsed:
                        jsonld_dates.append(
                            (key, parsed)
                        )

    # Prefer dateModified
    for key in [
        "dateModified",
        "dateReviewed",
        "datePublished",
    ]:

        for found_key, found_date in jsonld_dates:

            if found_key == key:
                return found_date

    # --------------------------------------------------------
    # 2. Meta tags
    # --------------------------------------------------------

    meta_candidates = [

        # OpenGraph
        ("property", "article:modified_time"),
        ("property", "article:published_time"),

        # Common metadata
        ("name", "dateModified"),
        ("name", "datePublished"),
        ("name", "date"),
        ("name", "last-modified"),
        ("name", "last_updated"),
        ("name", "updated"),

    ]

    for attribute, value in meta_candidates:

        tag = soup.find(
            "meta",
            attrs={attribute: value}
        )

        if tag:

            content = tag.get("content")

            parsed = normalize_date(content)

            if parsed:
                return parsed

    # --------------------------------------------------------
    # 3. <time datetime="">
    # --------------------------------------------------------

    time_tags = soup.find_all("time")

    for tag in time_tags:

        datetime_value = tag.get("datetime")

        if datetime_value:

            parsed = normalize_date(
                datetime_value
            )

            if parsed:
                return parsed

    # --------------------------------------------------------
    # 4. Visible text patterns
    # --------------------------------------------------------

    text = soup.get_text(
        " ",
        strip=True
    )

    patterns = [

        # Examples:
        # Last reviewed: January 12, 2025
        # Last reviewed January 12, 2025
        r"last\s+reviewed(?:\s+on)?[:\s]+"
        r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})",

        # Reviewed on January 12, 2025
        r"reviewed\s+on[:\s]+"
        r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})",

        # Last updated: January 12, 2025
        r"last\s+updated(?:\s+on)?[:\s]+"
        r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})",

        # Page last updated: January 12, 2025
        r"page\s+last\s+updated[:\s]+"
        r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})",

        # Updated: January 12, 2025
        r"\bupdated[:\s]+"
        r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})",

        # Published: January 12, 2025
        r"\bpublished[:\s]+"
        r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            parsed = normalize_date(
                match.group(1)
            )

            if parsed:
                return parsed

    # --------------------------------------------------------
    # No reliable date found
    # --------------------------------------------------------

    return None


# ============================================================
# Fetch + chunk
# ============================================================

def fetch_and_chunk(url, disease_tag):

    if not robots_allowed(url):

        print(
            f"Skipping (robots.txt disallows): {url}"
        )

        return []

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # --------------------------------------------------------
    # Extract real source date
    # --------------------------------------------------------

    source_date = extract_real_review_date(
        soup
    )

    if source_date:

        print(
            f"  Source date: {source_date}"
        )

    else:

        print(
            "  Source date: NOT FOUND"
        )

    # --------------------------------------------------------
    # Remove irrelevant page elements
    # --------------------------------------------------------

    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
        ]
    ):

        element.decompose()

    # --------------------------------------------------------
    # Extract paragraphs
    # --------------------------------------------------------

    paragraphs = [

        p.get_text(
            " ",
            strip=True
        )

        for p in soup.find_all("p")

    ]

    paragraphs = [

        p

        for p in paragraphs

        if len(p.split()) >= MIN_CHUNK_WORDS

    ]

    # --------------------------------------------------------
    # Chunk paragraphs
    # --------------------------------------------------------

    chunks = []

    for paragraph in paragraphs:

        words = paragraph.split()

        for start in range(
            0,
            len(words),
            MAX_CHUNK_WORDS
        ):

            chunk_words = words[
                start:start + MAX_CHUNK_WORDS
            ]

            if len(chunk_words) < MIN_CHUNK_WORDS:
                continue

            chunk_text = " ".join(
                chunk_words
            )

            chunk_id = hashlib.md5(
                (
                    url
                    + disease_tag
                    + chunk_text
                ).encode(
                    "utf-8"
                )
            ).hexdigest()[:12]

            chunks.append({

                "id": chunk_id,

                "text": chunk_text,

                "disease_tag": disease_tag,

                "source_url": url,

                "source_date": source_date,

            })

    return chunks


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Building guideline corpus - Version 3")
    print("=" * 60)

    seed_df = pd.read_csv(
        "data/disease_url_map.csv"
    )

    all_chunks = []

    for _, row in seed_df.iterrows():

        disease_tag = str(
            row["disease_tag"]
        ).strip()

        url = row["url"]

        if pd.isna(url):

            print(
                f"\nSkipping {disease_tag}: "
                "missing URL"
            )

            continue

        url = str(url).strip()

        if not url:

            print(
                f"\nSkipping {disease_tag}: "
                "empty URL"
            )

            continue

        print("\n" + "-" * 60)

        print(
            f"Disease: {disease_tag}"
        )

        print(
            f"URL: {url}"
        )

        try:

            chunks = fetch_and_chunk(
                url,
                disease_tag
            )

            all_chunks.extend(
                chunks
            )

            dates = sorted({

                c["source_date"]

                for c in chunks

                if c["source_date"]

            })

            if dates:

                print(
                    f"Extracted dates: {dates}"
                )

            else:

                print(
                    "WARNING: no reliable "
                    "source date found."
                )

            print(
                f"Chunks: {len(chunks)}"
            )

        except Exception as e:

            print(
                f"FAILED: {url}"
            )

            print(
                f"Error: {e}"
            )

        # Politeness delay
        time.sleep(2)

    # --------------------------------------------------------
    # Remove duplicate IDs
    # --------------------------------------------------------

    unique_chunks = {}

    for chunk in all_chunks:

        unique_chunks[
            chunk["id"]
        ] = chunk

    all_chunks = list(
        unique_chunks.values()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        "data/guideline_corpus.jsonl"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in all_chunks:

            f.write(

                json.dumps(
                    chunk,
                    ensure_ascii=False
                )

                + "\n"

            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    corpus_df = pd.DataFrame(
        all_chunks
    )

    print("\n" + "=" * 60)
    print("Corpus build complete")
    print("=" * 60)

    print(
        f"Total chunks     : {len(all_chunks)}"
    )

    if not corpus_df.empty:

        print(
            f"Unique diseases  : "
            f"{corpus_df['disease_tag'].nunique()}"
        )

        print(
            f"Unique URLs      : "
            f"{corpus_df['source_url'].nunique()}"
        )

        dated = corpus_df[
            corpus_df["source_date"].notna()
        ]

        print(
            f"Dated documents  : "
            f"{len(dated)}"
        )

        print(
            f"Undated documents: "
            f"{len(corpus_df) - len(dated)}"
        )

        print(
            f"Unique dates     : "
            f"{dated['source_date'].nunique()}"
        )

        if not dated.empty:

            print("\nDate distribution:")

            print(
                dated[
                    "source_date"
                ].value_counts().sort_index()
            )

    print(
        f"\nSaved -> {output_path}"
    )


if __name__ == "__main__":
    main()

