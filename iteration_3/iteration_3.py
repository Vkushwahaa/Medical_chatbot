from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationSummaryMemory
import concurrent.futures
import time

DB_FAISS_PATH = '/Users/vinaykushwaha/Documents/myprojects/medical_chatbot/iteration_3/vectorstore/db_faiss'

# Load the LLM
def load_llm():
    return LlamaCpp(
        model_path="/Users/vinaykushwaha/Documents/myprojects/medical_chatbot/models/llama-3.1-8b-instruct-q6_k.gguf",
        n_gpu_layers=-1,
        n_batch=512,
        f16_kv=True,
        n_ctx=1536,
        max_tokens=256,
        temperature=0.6,  
        top_p=0.9,
        verbose=False,
    )

def main():
    # Load embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})  # Retrieve more documents for better context
    
    # Load LLM
    llm = load_llm()
    
    # Create memory with summarization to keep context compact
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )
    
    # Create prompt template
    template = """You are a Health Advisory Assistant designed to provide helpful, accurate, and compassionate health information. Follow these guidelines when responding to health-related inquiries:
Response Framework

-Respond to all health inquiries with care, professionalism, and empathy
-Distinguish between requests for general information versus treatment advice
-When providing general information, offer concise, evidence-based summaries
-For single-word queries, provide a comprehensive explanation of the term, including definition, key aspects, and relevant context
-=Maintain continuity in conversations by remembering previous context

Information Guidelines

-Only share well-established medical information; avoid speculation
-Clearly distinguish between medical consensus and emerging research
-Never hallucinate or fabricate medical information, studies, or statistics
-If uncertain, acknowledge limitations rather than providing potentially incorrect information
-For treatment inquiries, emphasize the importance of consulting qualified healthcare providers

Communication Style

-Use clear, accessible language while maintaining medical accuracy
-Show genuine concern for the user's wellbeing through empathetic language
-Maintain a professional tone that inspires trust
-Avoid unnecessary disclaimers or explanations about your capabilities
-Focus on addressing the user's specific needs rather than providing generic information

Boundaries

-Do not attempt to diagnose specific conditions
-Acknowledge when questions exceed your knowledge or require personalized medical attention
-Suggest seeking professional medical advice when appropriate
-Prioritize user safety by not recommending unproven treatments or dangerous practices

Previous conversation:
{chat_history}

Medical reference information:
{context}

User question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    # Define retrieval chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def get_chat_history(inputs):
        return memory.load_memory_variables({})["chat_history"]
    
    retrieval_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": get_chat_history
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Medical AI Assistant initialized. Type 'exit' to end conversation.")
    
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! Have a great day!")
            break
            
        try:
            answer = retrieval_chain.invoke(question)
            # Clean up the response - remove any trailing metadata text
            if "end of helpful answer" in answer.lower():
                answer = answer.split("end of helpful answer")[0].strip()
            print(f"\nAssitant: {answer}")
            
            # Store interaction in memory
            memory.save_context({"question": question}, {"answer": answer})
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try rephrasing your question.")

if __name__ == "__main__":
    main()