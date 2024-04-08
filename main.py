from scraping import scrape_google_search_results, scrape_and_parse
from processing import split_document_content, create_vector_store_from_texts
from retrieval_qa import perform_query_with_retrieval, generate_response_from_context
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    query = "What are some upcoming Hindu festivals in 2024?"
    urls = scrape_google_search_results(query)
    documents = [scrape_and_parse(url) for url in urls if url.startswith("http://") or url.startswith("https://")]
    
    texts = []
    for document in documents:
        chunks = split_document_content(document)
        texts.extend(chunks)
    
    search = create_vector_store_from_texts(texts)
    qa_query = "When is Diwali in 2024?"
    retrieval_result = perform_query_with_retrieval(qa_query, search)
    response = generate_response_from_context(retrieval_result, qa_query)
    print(response)

if __name__ == "__main__":
    main()
