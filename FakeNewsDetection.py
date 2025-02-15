import spacy
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "6d387c1eef2d4a1da0c9c376b59f8319"
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    doc = nlp(text)
    keywords = [ent.text for ent in doc.ents]
    return keywords

def get_news_from_api(keywords, num_results=5):
    query = " ".join(keywords)
    sources = "the-times-of-india,the-verge,the-washington-post,techcrunch,the-wall-street-journal"
    url = f"https://newsapi.org/v2/everything?q={query}&sources={sources}&apiKey={API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [(art["title"], art["url"], art["description"]) for art in articles[:num_results]]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news from API: {e}")
        return []

def extract_article_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        article_text = " ".join([p.text for p in paragraphs])
        return article_text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""

def check_news_similarity(input_news, news_articles):
    vectorizer = TfidfVectorizer(stop_words="english")
    all_texts = [input_news] + news_articles
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarity_scores

def detect_fake_news(news_text):
    keywords = extract_keywords(news_text)
    news_articles = get_news_from_api(keywords)
    news_contents = [extract_article_content(article[1]) for article in news_articles]
    
    if not news_contents:
        return "No relevant news articles found. Cannot determine authenticity."

    similarity_scores = check_news_similarity(news_text, news_contents)
    threshold = 0.01
    avg_similarity = max(similarity_scores)
    return "Likely Real News" if avg_similarity > threshold else "Possibly Fake News"

if __name__ == "__main__":
    news_texts = ["Donald Trump elected as a new president of USA",
                  "No Tax for Income under 12 Lakhs for Individuals in India",
                  "India and US set a target of USD 500 Billion in bilateral trade by 2030",
                  "India withdraws high commisioner to canada amid escalating diplomatic row",
                  "Elon Musk's gift to PM Narendra Modi is likely a poece of the world's largest and most powerful rockert Starship.",
                  "Nita Ambani visits Harvard University as a Guest Speaker.",
                  "Protest took over at IIT Kanpur.",
                  "Alien invasion in USA",
                  "UFO found near England",
                  "Tsunami at the banks of Canada",
                  "Hawaii island is soon to be flooded"]
    for news_text in news_texts:
        result = detect_fake_news(news_text)
        print(result)
