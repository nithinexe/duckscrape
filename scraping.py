import requests
from bs4 import BeautifulSoup
import numpy as np
from urllib.parse import unquote
from readabilipy import simple_json_from_html_string
from langchain.schema import Document

def scrape_google_search_results(query):
    """ Perform a Google search and return cleaned URLs from the search results. """
    response = requests.get(f"https://www.google.com/search?q={query}")
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a")
    urls = []

    for link in [link for link in links if link["href"].startswith("/url?q=")]:
        url = link["href"].replace("/url?q=", "")
        url = unquote(url.split("&sa=")[0])
        if 'google.com/' in url or url.endswith('.pdf'):
            continue
        url = url.split('#')[0]
        urls.append(url)

    urls = list(np.unique(urls))
    if urls:
        urls.pop(0)  # Remove the first URL if it's not relevant or an ad
    
    return urls

def scrape_and_parse(url):
    """ Scrape a webpage and parse it into a Document object. """
    req = requests.get(url)
    article = simple_json_from_html_string(req.text, use_readability=True)
    plain_texts = [a['text'] for a in article['plain_text']] if article['plain_text'] else []
    page_content = plain_texts[0] if plain_texts else ""
    if len(plain_texts) > 1:
        page_content = '\n\n'.join(plain_texts)
    filtered_metadata = {}  # Simplified for demonstration
    return Document(page_content=page_content, metadata=filtered_metadata)

