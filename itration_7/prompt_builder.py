"""
Prompt Builder Module
Creates prompt templates for the LLM
"""

from langchain.prompts import ChatPromptTemplate

class PromptBuilder:
    def __init__(self):
        """Initialize the prompt builder with default templates"""
        self.default_template = """You are a Health Advisory Assistant designed to provide helpful, accurate, and compassionate health information. Follow these guidelines when responding to health-related inquiries, if the question is not "health" related simply say "I cant Answer That,please ask another question":

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
    
    def create_prompt(self, template_text=None):
        """
        Create a prompt template from text
        
        Args:
            template_text (str, optional): Custom template text. Defaults to None.
            
        Returns:
            ChatPromptTemplate: A langchain prompt template
        """
        if template_text is None:
            template_text = self.default_template
            
        return ChatPromptTemplate.from_template(template_text)
    
    def format_prompt(self, prompt_template, context, conversation_history, question):
        """
        Format a prompt with the given inputs
        
        Args:
            prompt_template (ChatPromptTemplate): The prompt template
            context (str): Retrieved document content
            conversation_history (str): Conversation history
            question (str): User question
            
        Returns:
            str: Fully formatted prompt
        """
        # Prepare the input for the prompt
        prompt_input = {
            "context": context,
            "conversation_history": conversation_history,
            "question": question
        }
        
        # Format the prompt with our inputs
        return prompt_template.format_prompt(**prompt_input).to_string()