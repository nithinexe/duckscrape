from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import OPENAI_API_KEY

def split_document_content(document):
    """ Splits document content into smaller chunks for processing. """
    text_chunks = CharacterTextSplitter(separator=' ', chunk_size=1000, chunk_overlap=200)
    texts = text_chunks.split_documents([document])
    return texts

def create_vector_store_from_texts(texts):
    """ Converts text chunks into embeddings and creates a vector store. """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    search = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"})
    return search

 