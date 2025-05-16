"""
Conversation Manager Module
Manages conversation history
"""

from datetime import datetime

class ConversationManager:
    def __init__(self):
        """Initialize the conversation manager"""
        self.conversation_history = []
        self.start_time = datetime.now()
        
    def add_exchange(self, user_input, assistant_response):
        """
        Add a conversation exchange to history
        
        Args:
            user_input (str): The user's input
            assistant_response (str): The assistant's response
        """
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response
        }
        self.conversation_history.append(exchange)
        
    def get_formatted_history(self, max_turns=5):
        """
        Return the conversation history formatted for display
        
        Args:
            max_turns (int): Maximum number of conversation turns to include
            
        Returns:
            str: Formatted conversation history
        """
        recent_history = self.conversation_history[-max_turns:] if len(self.conversation_history) > max_turns else self.conversation_history
        formatted_history = ""
        for exchange in recent_history:
            formatted_history += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
        return formatted_history
    
    def get_history_for_context(self, max_turns=5):
        """
        Return the conversation history formatted for the prompt
        
        Args:
            max_turns (int): Maximum number of conversation turns to include
            
        Returns:
            str: Conversation history formatted for prompt context
        """
        recent_history = self.conversation_history[-max_turns:] if len(self.conversation_history) > max_turns else self.conversation_history
        history_text = []
        for exchange in recent_history:
            history_text.append(f"User: {exchange['user']}")
            history_text.append(f"Assistant: {exchange['assistant']}")
        return "\n\n".join(history_text)
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        self.start_time = datetime.now()