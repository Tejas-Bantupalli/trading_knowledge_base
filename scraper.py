# scraper_async.py

import json
import os
import asyncio
from helpers import (
    extract_text_from_pdf,
    analyze_paper_with_ollama,  # NOW using Ollama
    safe_parse_json
)

OUTPUT_FILE = "arxiv_analysis_final.jsonl"
CONCURRENCY = 10

def load_processed_ids(output_path):
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_ids.add(entry["id"])
                except json.JSONDecodeError:
                    continue
    return processed_ids

def save_result_jsonl(output_path, entry):
    with open(output_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

async def process_paper(paper, semaphore):
    paper_id = paper["id"]
    async with semaphore:
        print(f"\n📄 Processing paper ID: {paper_id}")
        try:
            text = extract_text_from_pdf(paper["pdf_url"])
            if not text.strip():
                print(f"⚠️ Skipping {paper_id} — empty text.")
                return None

            print("🔍 Sending prompt to Ollama (phi3)...")
            raw_response = await analyze_paper_with_ollama(text)
            print("✅ Response received.")

            parsed = safe_parse_json(raw_response)
            if "error" in parsed:
                print(f"⚠️ Invalid JSON for {paper_id}")
                return None

            result = {
                "id": paper_id,
                "title": paper.get("title"),
                "analysis": parsed,
                "source_urls": {
                    "abs": paper.get("abs_url"),
                    "pdf": paper.get("pdf_url")
                }
            }

            save_result_jsonl(OUTPUT_FILE, result)
            print(f"💾 Saved result for {paper_id}")
            return result

        except Exception as e:
            print(f"❌ Error processing {paper_id}: {e}")
            return None

async def main_async():
    with open("arxiv_qfin_index.json") as f:
        papers = json.load(f)

    processed_ids = load_processed_ids(OUTPUT_FILE)
    to_process = [p for p in papers if p["id"] not in processed_ids]

    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [process_paper(paper, semaphore) for paper in to_process]

    await asyncio.gather(*tasks)

def main():
    asyncio.run(main_async())
    print("\n🎉 All done!")

if __name__ == "__main__":
    main()
