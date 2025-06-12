import os
import csv
from jinja2 import Template
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
# ────── 初始化 OpenRouter 客户端 ──────
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ────── 常量配置 ──────
DATA_FILE = "data/titles.csv"
OUTLINE_TEMPLATE = "prompts/outline_template.txt"
ARTICLE_TEMPLATE = "prompts/article_template.txt"
OUTPUT_DIR = "outputs"
TARGET_WORDS = 2000
MAX_TOKENS = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ────── 工具函数 ──────

def chat(prompt, temperature=0.7, max_tokens=MAX_TOKENS):
    """统一的对话请求函数"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def load_titles():
    """加载 CSV 文件中的所有行，返回字典列表"""
    with open(DATA_FILE, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def get_main_keyword(title_row: dict) -> str:
    """
    从 title_row（CSV 的一行字典）中直接提取 key_words 字段
    """
    return title_row.get('key_words', '').strip().lower()

def web_search_summary(keyword: str) -> str:
    prompt = f"Search the web and summarize recent content related to: {keyword}"
    return chat(prompt, temperature=0.3)

def generate_outline(summary: str, keyword: str) -> str:
    with open(OUTLINE_TEMPLATE, encoding='utf-8') as f:
        template = Template(f.read())
    prompt = template.render(summary=summary, keyword=keyword)
    return chat(prompt, temperature=0.3)
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
    rows = load_titles()
    for idx, row in enumerate(rows, start=1):
        article_id = str(idx)  # ✅ 添加这个唯一 ID
        keyword = get_main_keyword(row)
        print(f"\n🔍 Processing: {row['title']}")
        
        summary = web_search_summary(keyword)
        print(f"the keyword is {keyword}")
        print("✅ Summary done.")
        
        outline = generate_outline(summary, keyword)
        print("✅ Outline generated.")
        
        article = generate_article(row['title'], keyword, outline, summary)
        save_article(article_id, article)
        print(f"length: {len(article.split())} words")
        print(f"✅ Article saved to outputs/article-{article_id.zfill(3)}.md")

if __name__ == "__main__":
    main()

    
'''
def first_draft_article(title: str, keyword: str, outline: str, summary: str) -> str:
    with open(ARTICLE_TEMPLATE, encoding="utf-8") as f:
        template = Template(f.read())
    prompt = template.render(
        title=title,
        main_keyword=keyword,
        outline=outline,
        search_summary=summary
    )
    return chat(prompt, temperature=0.7)
def continue_article(existing: str, title: str) -> str:
    prompt = (
        f"The article titled '{title}' is currently {len(existing.split())} words. "
        f"Carefully expand only the existing sections of the article that are less than 150 words. Keep the original paragraph structure, order, tone, and intent unchanged. Do not introduce new paragraphs, rearrange content, or continue the article. Simply enrich short paragraphs with more depth, examples, or clarity. Continue this process only until the total word count of the article exceeds {TARGET_WORDS} words.Ensure that the outline and section headings remain unchanged. Do not split or merge paragraphs. Only add to short paragraphs to better meet the expected depth and coverage."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Here is the article so far:"},
            {"role": "user", "content": existing},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2048
    )
    return response.choices[0].message.content.strip()

def save_md(article_id: str, text: str):
    path = os.path.join(OUTPUT_DIR, f"article-{article_id.zfill(3)}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ────── 主流程 ──────

def main():
    titles = load_titles()
    for row in titles:
        title = row["title"]
        article_id = row["id"]
        keyword = get_main_keyword(title)

        print(f"\n🔍 Processing: {title}")

        summary = web_search_summary(keyword)
        print("✅ Summary done.")

        outline = generate_outline(summary, keyword)
        print("✅ Outline generated.")

        article = first_draft_article(title, keyword, outline, summary)
        print(f"   ↳ Draft length: {len(article.split())} words")

        while len(article.split()) < TARGET_WORDS:
            extra = continue_article(article, title)
            article += "\n\n" + extra
            print(f"   ↳ Extended to {len(article.split())} words")

        save_md(article_id, article)
        print("✅  Saved →", f"outputs/article-{article_id.zfill(3)}.md")

if __name__ == "__main__":
    main()
'''