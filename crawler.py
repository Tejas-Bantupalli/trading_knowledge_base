import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from time import sleep

def generate_months(start="2008-01"):
    start_date = datetime.strptime(start, "%Y-%m")
    today = datetime.today().replace(day=1)
    months = []
    while start_date <= today:
        months.append(start_date.strftime("%Y-%m"))
        start_date += timedelta(days=32)
        start_date = start_date.replace(day=1)
    return months

def crawl_arxiv_qfin():
    base_url = "https://arxiv.org/list/q-fin/"
    all_papers = []

    for month in generate_months():
        print(f"Crawling {month}...")
        url = f"{base_url}{month}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        dl = soup.find("dl", {"id": "articles"})
        if not dl:
            continue

        for dt in dl.find_all("dt"):
            abs_link = dt.find("a", title="Abstract")
            if not abs_link:
                continue

            paper_id = abs_link.get("id")
            abs_url = f"https://arxiv.org{abs_link.get('href')}"
            pdf_tag = dt.find("a", title="Download PDF")
            pdf_url = f"https://arxiv.org{pdf_tag.get('href')}" if pdf_tag else None

            all_papers.append({
                "id": paper_id,
                "abs_url": abs_url,
                "pdf_url": pdf_url,
                "month": month
            })

        sleep(1)  # polite delay

    return all_papers

def main():
    papers = crawl_arxiv_qfin()
    print(f"Found {len(papers)} papers.")
    for paper in papers:
        print(f"ID: {paper['id']}, Abstract URL: {paper['abs_url']}, PDF URL: {paper['pdf_url']}, Month: {paper['month']}")
    import json

    with open("arxiv_qfin_index.json", "w") as f:
        json.dump(papers, f, indent=2)

    print(f"Saved {len(papers)} papers to arxiv_qfin_index.json ✅")

if __name__ == "__main__":
    main()