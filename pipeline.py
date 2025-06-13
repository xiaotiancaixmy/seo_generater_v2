import os
import csv
from jinja2 import Template
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
import re

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
SIMILARITY_THRESHOLD = 0.8  # 相似度阈值

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
    prompt = f"""Search the web and provide a comprehensive summary of recent content related to: {keyword}

Requirements:
1. First list the top 3-5 most relevant web sources you found
2. Then provide a detailed summary of the information from these sources
3. Include key statistics, trends, and insights
4. Focus on recent and authoritative information
5. Format the output as follows:

SOURCES:
- [Source 1]
- [Source 2]
- [Source 3]

SUMMARY:
[Your detailed summary here]"""
    return chat(prompt, temperature=0.3)

def generate_outline(summary: str, keyword: str) -> str:
    with open(OUTLINE_TEMPLATE, encoding='utf-8') as f:
        template = Template(f.read())
    prompt = template.render(summary=summary, keyword=keyword)
    return chat(prompt, temperature=0.3)

def check_content_duplication(text: str) -> tuple[bool, list[str]]:
    """检查内容重复
    返回: (是否有重复, 重复内容列表)
    """
    # 将文本分成段落
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # 检查段落级别的重复
    duplicates = []
    for i, p1 in enumerate(paragraphs):
        for j, p2 in enumerate(paragraphs[i+1:], i+1):
            # 计算两个段落的相似度
            similarity = calculate_similarity(p1, p2)
            if similarity > SIMILARITY_THRESHOLD:
                duplicates.append(f"段落 {i+1} 和 {j+1} 相似度: {similarity:.2f}")
    
    return len(duplicates) > 0, duplicates

def calculate_similarity(text1: str, text2: str) -> float:
    """计算两段文本的相似度"""
    # 使用简单的词频比较
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def optimize_content(text: str) -> str:
    """优化内容，去除重复并重写"""
    has_duplicates, duplicates = check_content_duplication(text)
    
    if has_duplicates:
        print("⚠️ 检测到内容重复，正在优化...")
        prompt = f"""
        以下文章存在内容重复，请重写以消除重复，同时保持文章的整体结构和质量：
        
        {text}
        
        重复内容：
        {chr(10).join(duplicates)}
        
        要求：
        1. 保持文章的主要观点和信息
        2. 使用不同的表达方式重写重复的部分
        3. 确保文章流畅自然
        4. 保持原有的段落结构
        5. 不要改变文章的主题和目的
        """
        return chat(prompt, temperature=0.7)
    
    return text

def first_draft_article(title: str, keyword: str, outline: str, summary: str) -> str:
    """生成文章初稿"""
    with open(ARTICLE_TEMPLATE, encoding="utf-8") as f:
        template = Template(f.read())
    prompt = template.render(
        title=title,
        main_keyword=keyword,
        outline=outline,
        search_summary=summary
    )
    return chat(prompt, temperature=0.7)

def expand_section(section: str, target_words: int) -> str:
    """扩写特定段落"""
    prompt = f"""
    Expand the following section to approximately {target_words} words while maintaining its original meaning and structure:
    
    {section}
    
    Requirements:
    - Keep the same tone and style
    - Add relevant examples and details
    - Maintain paragraph structure
    - Ensure natural flow
    - Avoid repeating information
    """
    return chat(prompt, temperature=0.7)

def ensure_faq_section(article: str) -> str:
    """确保文章包含FAQ部分"""
    if "FAQ" not in article and "Frequently Asked Questions" not in article:
        prompt = f"""
        Add a comprehensive FAQ section to this article with 5-7 common questions and detailed answers:
        
        {article}
        
        Requirements:
        - Add 5-7 relevant questions
        - Provide detailed, helpful answers
        - Use natural language
        - Include examples where appropriate
        - Ensure questions are unique and not repetitive
        """
        return chat(prompt, temperature=0.7)
    return article

def save_article(article_id: str, content: str):
    """保存文章到文件"""
    path = os.path.join(OUTPUT_DIR, f"article-{article_id.zfill(3)}.md")
    with open(path, "w", encoding='utf-8') as f:
        f.write(content)

def main():
    rows = load_titles()
    for idx, row in enumerate(rows, start=1):
        article_id = str(idx)
        keyword = get_main_keyword(row)
        print(f"\n🔍 Processing: {row['title']}")
        
        # 1. 生成摘要和提纲
        summary = web_search_summary(keyword)
        print(f"the keyword is {keyword}")
        print("✅ Summary done.")
        
        outline = generate_outline(summary, keyword)
        print(f"the outline is {outline}")
        print("✅ Outline generated.")
        
        # 2. 生成初稿
        article = first_draft_article(row['title'], keyword, outline, summary)
        print(f"   ↳ Draft length: {len(article.split())} words")
        
        # 3. 确保文章长度
        while len(article.split()) < TARGET_WORDS:
            # 分析文章结构，找出最短的部分进行扩写
            sections = article.split('\n\n')
            shortest_section = min(sections, key=lambda x: len(x.split()))
            expanded_section = expand_section(shortest_section, 300)
            article = article.replace(shortest_section, expanded_section)
            print(f"   ↳ Extended to {len(article.split())} words")
        
        # 4. 确保包含FAQ部分
        article = ensure_faq_section(article)
        print("✅ FAQ section added/verified")
        
        # 5. 检查并优化重复内容
        article = optimize_content(article)
        print("✅ Content optimized")
        
        # 6. 保存文章
        save_article(article_id, article)
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