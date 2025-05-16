"""
Main Application Module
Entry point for the RAG-powered medical assistant chatbot
"""

import time
from dotenv import load_dotenv

# Import modules
from llm_handler import load_llm
from retriever import Retriever
from conversation import ConversationManager
from prompt_builder import PromptBuilder
from utils import with_timeout, print_help

# Load environment variables
load_dotenv()

def main():
    """Main function to run the chatbot"""
    print("Initializing RAG system...")
    
    try:
        # Initialize components
        retriever = Retriever(k=2)
        conversation_manager = ConversationManager()
        prompt_builder = PromptBuilder()
        llm = load_llm()
        
        # Create prompt template
        prompt_template = prompt_builder.create_prompt()
        
        print("RAG system initialized successfully!")
        print_help()
        
        # Main conversation loop
        while True:
            user_input = input("\nYou: ").strip()
            
            # Command handling
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye! Have a great day!")
                break
            elif user_input.lower() == "help":
                print_help()
                continue
            elif user_input.lower() == "clear":
                conversation_manager.clear_history()
                print("Conversation history cleared")
                continue
                
            try:
                # Get conversation history for context
                history = conversation_manager.get_history_for_context()
                
                print(f"\nProcessing query: '{user_input}'")
                
                # Define function to get answer
                def get_answer():
                    # Retrieve relevant documents
                    docs = retriever.get_relevant_documents(user_input)
                    context = retriever.format_documents(docs)
                    
                    # Format prompt
                    formatted_prompt = prompt_builder.format_prompt(
                        prompt_template, 
                        context, 
                        history, 
                        user_input
                    )
                    
                    # Get response from LLM
                    return llm.invoke(formatted_prompt)
                
                # Get answer with timeout protection
                start_time = time.time()
                raw_answer = with_timeout(get_answer, timeout_duration=30)
                answer = str(raw_answer.content if hasattr(raw_answer, 'content') else raw_answer)
                
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