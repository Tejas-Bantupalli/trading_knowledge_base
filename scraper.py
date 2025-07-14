# scraper.py
import json
import time
from helpers import extract_text_from_pdf, analyze_paper_with_gemini, safe_parse_json


def main():
    with open("arxiv_qfin_index.json") as f:
        papers = json.load(f)

    results = []
    processed_ids = set()

    for paper in papers:
        if paper["id"] in processed_ids:
            continue

        print(f"Processing paper ID: {paper['id']}")
        try:
            text = extract_text_from_pdf(paper["pdf_url"])
            # Step 2: Analyze with Gemini
            analysis_raw = analyze_paper_with_gemini(text)
            analysis = safe_parse_json(analysis_raw)

            if "error" in analysis:
                print(f"⚠️ Invalid JSON for {paper['id']}:")
                print("🔸Raw:\n", analysis["raw_response"][:500])  # optional: limit output
                continue
                print(f"⚠️ Invalid JSON for {paper_id}:")
                print("🔸Raw:\n", analysis["raw_response"][:500])  # optional: limit output
                continue


            results.append({
                "id": paper["id"],
                "title": paper.get("title"),
                "analysis": analysis,
                "source_urls": {
                    "abs": paper.get("abs_url"),
                    "pdf": paper.get("pdf_url")
                }
            })

            processed_ids.add(paper["id"])

        except Exception as e:
            print(f"❌ Error processing {paper['id']}: {e}")
            continue

        time.sleep(1)  # optional throttle

    with open("arxiv_analysis_final.json", "w") as out_f:
        json.dump(results, out_f, indent=2)

    print(f"\n✅ Saved {len(results)} papers to arxiv_analysis_final.json")

if __name__ == "__main__":
    main()
