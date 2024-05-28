from langchain_community.llms import CTransformers
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.chains.retrieval_qa.base import RetrievalQA 
from langchain_core.prompts import PromptTemplate
import os
import dotenv

dotenv.load_dotenv()

DB_FAISS_PATH = 'vectorstore/db_faiss'
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
API_TOKEN = os.getenv('API_TOKEN')

custom_prompt_template = """ 
    Context: {context}
    Question: {question}
"""


def set_custom_prompt():
    """
    Set the custom prompt for the QA model
    """
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']  # Add input_variables here
    )
    return prompt

def load_llm():
    llm = HuggingFaceEndpoint(repo_id=model_name, huggingfacehub_api_token=API_TOKEN)
    return llm

def retrieval_qa_llm(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_llm(llm, qa_prompt, db)
    return qa

def final_result(query, chat_history):
    qa = qa_bot()
    context = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
    combined_context = f"{context}\nQ: {query}"
    
    result = qa({'query': combined_context})

    answer = result['result']
    source_documents = result['source_documents']
    combined_content = " ".join([doc.page_content.replace('\n', ' ') for doc in source_documents])

    output = f"""
    Query: {query}
    
    Answer: {answer}
    
    Source Documents Content:
    {combined_content}
    """
    return answer, output

if __name__ == "__main__":
    chat_history = []
    while True:
        query = input("Enter your query: ")
        answer, result_output = final_result(query, chat_history)
        print(result_output)
        chat_history.append((query, answer))



       