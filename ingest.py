from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"


def create_vectorstore_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)

    documents = loader.load()
    
    if not documents:
        print("Error: No documents loaded from", DATA_PATH)
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    texts = text_splitter.split_documents(documents)

    if not texts:
        print("Error: No text extracted from documents")
        return

    try:
        embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda'})
        db = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        print("Error creating vectorstore:", e)
        return

    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vectorstore_db()
