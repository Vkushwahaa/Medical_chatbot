from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Constants
DATA_PATH = '/Users/vinaykushwaha/Documents/myprojects/medical_chatbot/itration_7/data'
DB_FAISS_PATH = '/Users/vinaykushwaha/Documents/myprojects/medical_chatbot/itration_7/vectorstore/db_faiss'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def create_vector_db():
    """Create a FAISS vector store from PDF documents"""
    try:
        # Ensure data directory exists
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data directory not found at {DATA_PATH}")

        # Initialize document loader
        loader = DirectoryLoader(
            DATA_PATH,
            glob='*.pdf',
            loader_cls=PyPDFLoader
        )

        # Load and split documents
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )

        # Create and save vector store
        print(f"Creating vector store from {len(texts)} text chunks...")
        db = FAISS.from_documents(texts, embeddings)
        
        # Ensure vector store directory exists
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        db.save_local(DB_FAISS_PATH)
        print(f"Vector store saved to {DB_FAISS_PATH}")
        
        return db

    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

if __name__ == "__main__":
    create_vector_db()