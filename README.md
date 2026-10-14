# Fake News Detection — Article Similarity Prototype

This Python project explores a similarity-based approach to checking a news claim against articles retrieved from online news sources. It extracts named entities from the submitted text, searches for related articles, extracts article text, and compares the text using TF-IDF vectors and cosine similarity.

## How it works
1. spaCy identifies named entities that become search keywords.
2. NewsAPI is queried for potentially related articles.
3. Article pages are fetched and their paragraph text is extracted with BeautifulSoup.
4. TF-IDF and cosine similarity compare the submitted claim with retrieved article content.
5. The program reports a similarity-based result when relevant articles are available.

## Files
- `FakeNewsDetection.py` — main detection workflow.
- `FactChecking.ipynb` — notebook for experiments and analysis.

## Requirements
Python 3.9+ recommended, plus `requests`, `beautifulsoup4`, `scikit-learn`, and spaCy with the English model `en_core_web_sm`.

```bash
git clone https://github.com/Saikabilane/Fake-News-Detection.git
cd Fake-News-Detection
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install requests beautifulsoup4 scikit-learn spacy jupyter
python -m spacy download en_core_web_sm
```

## API configuration
Create a NewsAPI key and load it from an environment variable instead of hard-coding it in source. If a real key has been committed, revoke/rotate it and remove it from the current code; removing it from the latest commit does not erase it from Git history.

## Important limitations
Text similarity is not proof that a claim is true or false. Search coverage, publisher selection, article availability and paraphrasing can all affect results. Use this project as a learning prototype and verify claims with multiple reputable sources.

## License
No license file is currently included. Contact the repository owner for reuse permission.
