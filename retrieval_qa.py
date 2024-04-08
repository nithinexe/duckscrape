from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from openai import Client
from config import OPENAI_API_KEY

def perform_query_with_retrieval(qa_query, search):
    """ Performs a retrieval-based QA on the documents using a pre-initialized vector store. """
    model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.4)
    qa = RetrievalQA.from_chain_type(model, chain_type="stuff", return_source_documents=False, retriever=search.as_retriever())
    result = qa({"query": qa_query})
    return result

def generate_response_from_context(result, query):
    """ Generates a response based on the retrieved context and the user query. """
    client = Client(api_key=OPENAI_API_KEY)
    prompt_template = "Please answer the question with reference to the given context and scraped results"
    prompt = f"Document: \n{result}\nQuestion: {query}\n{prompt_template}"
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.5,
        max_tokens=550
    )
    
    return response.choices[0].text


