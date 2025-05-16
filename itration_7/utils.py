"""
Utility functions
"""

import threading
import time

def with_timeout(func, args=(), kwargs={}, timeout_duration=30):
    """
    Run a function with a timeout
    
    Args:
        func (callable): Function to run
        args (tuple): Function arguments
        kwargs (dict): Function keyword arguments
        timeout_duration (int): Timeout in seconds
        
    Returns:
        Any: Function result or timeout message
    """
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

def print_help():
    """Print help menu"""
    print("\n====== Medical Assistant Commands ======")
    print("- Type your medical question normally")
    print("- Type 'exit' to end the conversation")
    print("- Type 'clear' to clear conversation history")
    print("- Type 'help' to show this menu")
    print("========================================")