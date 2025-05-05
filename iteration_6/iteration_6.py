from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import time
import threading
import json
import os
import uuid
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()




CONVERSATIONS_PATH = os.environ.get("CONVERSATIONS_PATH", "conversations")
# Create conversations directory if it doesn't exist
os.makedirs(CONVERSATIONS_PATH, exist_ok=True)

DB_FAISS_PATH = os.environ.get("DB_FAISS_PATH")
if not DB_FAISS_PATH:
    raise ValueError("DB_FAISS_PATH environment variable not set")

API = os.environ.get("GEMINI_API")
if not API:
    raise ValueError("GEMINI_API environment variable not set")

# Load the LLM model
def load_llm():
    return ChatGoogleGenerativeAI(
        model = "gemini-2.0-flash",
        temperature=0.6,
        max_tokens=None,
        timeout=20,
        n_ctx=2000,
        max_retries=2,
        google_api_key=API
    )
# # Load the LLM model
# def load_llm():
#     return LlamaCpp(
#         model_path="/Users/vinaykushwaha/Documents/myprojects/medical_chatbot/models/gemma-3-4b-it-q4_0.gguf",
#         n_gpu_layers=-1,
#         n_batch=512,
#         f16_kv=True,
#         n_ctx=1536,
#         max_tokens=256,
#         temperature=0.6,  
#         top_p=0.9,
#         verbose=False,  # Set to True for verbose output
#     )

# Simple timeout function
def with_timeout(func, args=(), kwargs={}, timeout_duration=30):
    result = [None]
    error = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = e
            
    thread = threading.Thread(target=target)
    thread.daemon = True
    
    print(f"Starting processing with {timeout_duration} second timeout...")
    start_time = time.time()
    
    thread.start()
    thread.join(timeout_duration)
    
    elapsed_time = time.time() - start_time
    print(f"Processing took {elapsed_time:.2f} seconds")
    
    if thread.is_alive():
        print(f"WARNING: Operation timed out after {timeout_duration} seconds")
        return f"Response took too long (over {timeout_duration} seconds). Please try a simpler question."
    if error[0] is not None:
        print(f"ERROR: {str(error[0])}")
        return f"Error: {str(error[0])}"
    return result[0]

# Conversation management class
class ConversationManager:
    def __init__(self, save_dir=CONVERSATIONS_PATH):
        self.save_dir = save_dir
        self.conversation_id = str(uuid.uuid4())
        self.conversation_history = []
        self.start_time = datetime.now()
        
    def add_exchange(self, user_input, assistant_response):
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response
        }
        self.conversation_history.append(exchange)
        self._save_conversation()
        
    def get_formatted_history(self, max_turns=5):
        """Return the conversation history formatted for the context window"""
        recent_history = self.conversation_history[-max_turns:] if len(self.conversation_history) > max_turns else self.conversation_history
        formatted_history = ""
        for exchange in recent_history:
            formatted_history += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
        return formatted_history
    
    def get_history_for_context(self, max_turns=5):
        """Return the conversation history formatted for the prompt"""
        recent_history = self.conversation_history[-max_turns:] if len(self.conversation_history) > max_turns else self.conversation_history
        history_text = []
        for exchange in recent_history:
            history_text.append(f"User: {exchange['user']}")
            history_text.append(f"Assistant: {exchange['assistant']}")
        return "\n\n".join(history_text)
    
    def _save_conversation(self):
        """Save the current conversation to disk"""
        conversation_data = {
            "id": self.conversation_id,
            "start_time": self.start_time.isoformat(),
            "history": self.conversation_history
        }
        
        file_path = os.path.join(self.save_dir, f"conversation_{self.conversation_id}.json")
        with open(file_path, 'w') as f:
            json.dump(conversation_data, f, indent=2)
            
    def load_conversation(self, conversation_id):
        """Load a previous conversation"""
        file_path = os.path.join(self.save_dir, f"conversation_{conversation_id}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.conversation_id = data["id"]
                self.start_time = datetime.fromisoformat(data["start_time"])
                self.conversation_history = data["history"]
            return True
        return False
    
    def list_conversations(self):
        """List all saved conversations"""
        conversations = []
        for filename in os.listdir(self.save_dir):
            if filename.startswith("conversation_") and filename.endswith(".json"):
                file_path = os.path.join(self.save_dir, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    conversations.append({
                        "id": data["id"],
                        "start_time": data["start_time"],
                        "exchanges": len(data["history"])
                    })
        return conversations

def main():
    print("Initializing RAG system...")
    
    try:
        # Load embeddings
        print("Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load vector store
        print(f"Loading vector database from {DB_FAISS_PATH}...")
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
        print("Vector database loaded successfully")
        

        # Load LLM
        print("Loading LLM model...")
        llm = load_llm()
        print("LLM model loaded successfully")
        
        # Create conversation manager
        conversation_manager = ConversationManager()
        print(f"Created new conversation with ID: {conversation_manager.conversation_id}")
        
        # Create prompt template with conversation history
        template = """You are a Health Advisory Assistant designed to provide helpful, accurate, and compassionate health information. Follow these guidelines when responding to health-related inquiries, if the question is not "health" related simply say "I cant Answer That,please ask another question":

Response Framework:
- (do not repeat phrases such as "okay i understand","i understand",etc.)
- Respond to all health inquiries with care, professionalism, and empathy
- Distinguish between requests for general information versus treatment advice
- When providing general information, offer concise, evidence-based summaries
- For single-word queries, provide a comprehensive explanation of the term, including definition, key aspects, and relevant context
- Remember previous parts of the conversation and maintain continuity when answering follow-up questions
Information Guidelines:
- Only share well-established medical information; avoid speculation
- Clearly distinguish between medical consensus and emerging research
- Never hallucinate or fabricate medical information, studies, or statistics
- If uncertain, acknowledge limitations rather than providing potentially incorrect information
- For treatment inquiries, emphasize the importance of consulting qualified healthcare providers

Medical reference information:
{context}

Conversation history:
{conversation_history}

Current question: {question}

Answer conversationally while keeping track of the entire conversation context:"""

        prompt = ChatPromptTemplate.from_template(template)
        print("Prompt template created")
        
        # Function to format retrieved documents
        def format_docs(docs):
            formatted = "\n\n".join(doc.page_content for doc in docs)
            print(f"Retrieved {len(docs)} documents ({len(formatted)} characters)")
            return formatted
        
        # Set up the RAG pipeline
        print("Setting up RAG pipeline...")
        
        def get_conversation_chain(question, history):
            # Get relevant documents
            docs = retriever.invoke(question)
            formatted_docs = format_docs(docs)
            
            # Prepare the input for the prompt
            chain_input = {
                "context": formatted_docs,
                "conversation_history": history,
                "question": question
            }
            
            # Format the prompt with our inputs
            formatted_prompt = prompt.format_prompt(**chain_input).to_string()
            
            # Run the LLM on the formatted prompt
            llm_result = llm.invoke(formatted_prompt)
            
            # Return the result
            return llm_result
        
        print("RAG pipeline ready")
        
        # Print help menu
        def print_help():
            print("\n====== Medical Assistant Commands ======")
            print("- Type your medical question normally")
            print("- Type 'exit' to end the conversation")
            print("- Type 'save' to explicitly save the conversation")
            print("- Type 'list' to show saved conversations")
            print("- Type 'load <id>' to load a previous conversation")
            print("- Type 'help' to show this menu")
            print("========================================")
        
        print("\n====== RAG-Powered Medical Assistant initialized ======")
        print("Type 'help' to see available commands")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            # Command handling
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Saving conversation and exiting...")
                conversation_manager._save_conversation()
                print(f"Conversation saved with ID: {conversation_manager.conversation_id}")
                print("Goodbye! Have a great day!")
                break
            elif user_input.lower() == "help":
                print_help()
                continue
            elif user_input.lower() == "save":
                conversation_manager._save_conversation()
                print(f"Conversation saved with ID: {conversation_manager.conversation_id}")
                continue
            elif user_input.lower() == "list":
                conversations = conversation_manager.list_conversations()
                print("\n===== Saved Conversations =====")
                for convo in conversations:
                    print(f"ID: {convo['id']}")
                    print(f"Date: {convo['start_time']}")
                    print(f"Messages: {convo['exchanges']}")
                    print("-" * 30)
                continue
            elif user_input.lower().startswith("load "):
                convo_id = user_input[5:].strip()
                if conversation_manager.load_conversation(convo_id):
                    print(f"Loaded conversation {convo_id}")
                    print("\n----- Conversation History -----")
                    print(conversation_manager.get_formatted_history())
                    print("-" * 30)
                else:
                    print(f"Could not find conversation with ID {convo_id}")
                continue
                
            try:
                # Get conversation history for context
                history = conversation_manager.get_history_for_context()
                
                print(f"\nProcessing query: '{user_input}'")
                print("Retrieving relevant documents...")
                
                start_time = time.time()
                
                def get_answer():
                    return get_conversation_chain(user_input, history)
                
                raw_answer = with_timeout(get_answer, timeout_duration=30)
                answer = str(raw_answer.content if hasattr(raw_answer, 'content') else raw_answer)
                print(f"Answer type: {type(raw_answer)}")
                
                # Clean up answer if needed
                if isinstance(answer, str) and "end of helpful answer" in answer.lower():
                    answer = answer.split("end of helpful answer")[0].strip()
                
                # Add to conversation history
                conversation_manager.add_exchange(user_input, answer)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                print(f"\nAssistant: {answer}")
                print(f"Total response time: {total_time:.2f} seconds")
                
            except Exception as e:
                print(f"ERROR: {str(e)}")
                print("Please try rephrasing your question.")
    
    except Exception as e:
        print(f"CRITICAL ERROR during initialization: {str(e)}")
        print("Please check your file paths and model configuration.")

if __name__ == "__main__":
    main()