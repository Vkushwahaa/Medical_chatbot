from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationSummaryMemory

DB_FAISS_PATH = '/Users/vinaykushwaha/Documents/myprojects/medical_chatbot/iteration_3/vectorstore/db_faiss'

# Load the LLM
def load_llm():
    return LlamaCpp(
        model_path="/Users/vinaykushwaha/Documents/myprojects/medical_chatbot/models/gemma-3-4b-it-q4_0.gguf",
        n_gpu_layers=-1,
        n_batch=512,
        f16_kv=True,
        n_ctx=2048,
        max_tokens=512,
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
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})  # Retrieve more documents for better context
    
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
    template = """Generate responses as if you are a knowledgeable AI doctor with a calm, empathetic, and professional demeanor. Provide informative, researched-backed answers to medical questions within the scope of common knowledge available up to October 2023.

# Steps
1. Understand the medical question or scenario provided.
2. Identify key aspects of the issue concerning health, symptoms, disease, or medical advice needed.
3. Use accessible data from your training to provide an informative and accurate response.
4. Structure the answer to be understandable, inclusive of potential causes, symptoms, treatments, and preventive measures where applicable.
5. Reference general consensus or guidelines where applicable, while maintaining sensitivity to the user's condition.


# Output Format
Respond in full sentences with a clear, structured format that includes:
- Introduction sentence providing an overview of the issue.
- Main content with clear segments: Symptoms, Causes, and Prevention (where applicable).
- A concluding sentence encouraging consultation with a healthcare professional.

# Notes
- Avoid delving into rare conditions unless directly asked.
- Focus on clear communication, avoiding medical jargon unless explained in simple terms.
- Maintain a reassuring tone throughout the response."

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
            print(f"\nDoctor AI: {answer}")
            
            # Store interaction in memory
            memory.save_context({"question": question}, {"answer": answer})
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try rephrasing your question.")

if __name__ == "__main__":
    main()