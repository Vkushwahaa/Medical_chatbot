"""
LLM Handler Module
Responsible for loading and managing the LLM model
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check for API key
API = os.environ.get("GEMINI_API")
if not API:
    raise ValueError("GEMINI_API environment variable not set")

def load_llm():
    """Load and configure the LLM model"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.6,
        max_tokens=None,
        timeout=20,
        n_ctx=2000,
        max_retries=2,
        google_api_key=API
    )

# Alternative model loader commented out - can be used for local models
"""
def load_local_llm():
    from langchain_community.llms import LlamaCpp
    
    return LlamaCpp(
        model_path="path/to/your/model.gguf",
        n_gpu_layers=-1,
        n_batch=512,
        f16_kv=True,
        n_ctx=1536,
        max_tokens=256,
        temperature=0.6,  
        top_p=0.9,
        verbose=False,
    )
"""