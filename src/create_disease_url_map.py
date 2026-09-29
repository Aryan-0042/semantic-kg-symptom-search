import pandas as pd
import time
from ddgs import DDGS

OUTPUT = "data/disease_url_map.csv"


def search_official(disease):
    queries = [
        f'site:medlineplus.gov "{disease}"',
        f'site:cdc.gov "{disease}"'
    ]

    with DDGS() as ddgs:
        for q in queries:
            try:
                results = list(ddgs.text(q, max_results=5))

                for r in results:
                    url = r.get("href") or r.get("url")

                    if not url:
                        continue

                    if "medlineplus.gov" in url:
                        return url

                    if "cdc.gov" in url:
                        return url

            except Exception:
                pass

    return ""


def main():

    df = pd.read_csv("data/symptom_disease.csv")

    diseases = sorted(df["Disease"].unique())

    rows = []

    print(f"Searching {len(diseases)} diseases...\n")

    for disease in diseases:

        print(f"Searching: {disease}")

        url = search_official(disease)

        if url:
            print("  ✓", url)
        else:
            print("  ✗ Not Found")

        rows.append({
            "disease_tag": disease,
            "url": url
        })

        time.sleep(1)

    out = pd.DataFrame(rows)

    out.to_csv(OUTPUT, index=False)

    print("\nDone!")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()