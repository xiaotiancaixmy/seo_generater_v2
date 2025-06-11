import os
import csv
import json
from jinja2 import Template
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

DATA_FILE = "data/titles.csv"
OUTLINE_TEMPLATE = "prompts/outline_template.txt"
ARTICLE_TEMPLATE = "prompts/seo_template.txt"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_titles():
    with open(DATA_FILE, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def get_main_keyword(title: str) -> str:
    return title.split()[2].lower()

def web_search_summary(keyword: str) -> str:
    prompt = f"Search the web and summarize recent content related to: {keyword}"
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def generate_outline(summary: str, keyword: str) -> str:
    with open(OUTLINE_TEMPLATE, encoding='utf-8') as f:
        template = Template(f.read())
    prompt = template.render(summary=summary, keyword=keyword)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def generate_article(title: str, keyword: str, outline: str, summary: str) -> str:
    with open(ARTICLE_TEMPLATE, encoding='utf-8') as f:
        template = Template(f.read())
    prompt = template.render(title=title, main_keyword=keyword, outline=outline, search_summary=summary)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def save_article(article_id: str, content: str):
    path = os.path.join(OUTPUT_DIR, f"article-{article_id.zfill(3)}.md")
    with open(path, "w", encoding='utf-8') as f:
        f.write(content)

def main():
    titles = load_titles()
    for row in titles:
        title = row["title"]
        article_id = row["id"]
        keyword = get_main_keyword(title)

        print(f"\n🔍 Processing: {title}")

        # 1. 搜索相关内容
        summary = web_search_summary(keyword)
        print("✅ Summary done.")

        # 2. 基于搜索内容生成 outline
        outline = generate_outline(summary, keyword)
        print("✅ Outline generated.")

        # 3. 基于 outline + summary + title 生成完整文章
        article = generate_article(title, keyword, outline, summary)
        save_article(article_id, article)

        print(f"✅ Article saved to outputs/article-{article_id.zfill(3)}.md")

if __name__ == "__main__":
    main()