"""
Retriever Module
Handles loading and querying vector database
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get vector database path from environment
DB_FAISS_PATH = os.getenv("DB_FAISS_PATH")
if not DB_FAISS_PATH:
    raise ValueError("DB_FAISS_PATH environment variable not set")

print(f"Using vector store path: {DB_FAISS_PATH}")


class Retriever:
    def __init__(self, k=2):
        """
        Initialize the retriever with vector database
        
        Args:
            k (int): Number of documents to retrieve for each query
        """
        self.k = k
        self.vectorstore = None
        self.retriever = None
        self.load_vectorstore()
        
    def load_vectorstore(self):
        """Load the vector database from disk"""
        print(f"Loading vector database from {DB_FAISS_PATH}...")
        
        # Load embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load the vector database
        self.vectorstore = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={'k': self.k}
        )
        
        print("Vector database loaded successfully")
    
    def get_relevant_documents(self, query):
        """
        Retrieve relevant documents for a query
        
        Args:
            query (str): The user query
            
        Returns:
            list: List of retrieved documents
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized")
            
        return self.retriever.invoke(query)
    
    def format_documents(self, documents):
        """
        Format retrieved documents into a string
        
        Args:
            documents (list): List of document objects
            
        Returns:
            str: Formatted string of document contents
        """
        formatted = "\n\n".join(doc.page_content for doc in documents)
        print(f"Retrieved {len(documents)} documents ({len(formatted)} characters)")
        return formatted