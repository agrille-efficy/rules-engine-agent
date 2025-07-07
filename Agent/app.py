import os
import re
import signal
import sys
from langchain_core.messages import HumanMessage
from agent import build_graph


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Agent execution timed out after 60 seconds")


class BasicAgent:
    """A langgraph agent."""
    def __init__(self):
        print("🤖 BasicAgent initialized.")
        self.graph = build_graph()

    def __call__(self, question: str) -> str:
        print(f"🔍 Processing question: {question[:100]}...")
        
        # Set up timeout for Windows (if available)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout
        
        try:
            # Wrap the question in a HumanMessage from langchain_core
            messages = [HumanMessage(content=question)]
            messages = self.graph.invoke({"messages": messages})
            answer = messages['messages'][-1].content
            
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel timeout
            
            return answer
        except TimeoutError:
            return "❌ Agent execution timed out. This might be due to slow API responses or network issues."
        except Exception as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel timeout
            raise


def run_agent_cli(question_text: str) -> str:
    """Run the agent on a user-provided question"""
    if not question_text or not question_text.strip():
        return "❌ Please provide a question."
    
    try:
        agent = BasicAgent()
        response = agent(question_text.strip())
        
        # Extract final answer if it follows the expected format
        match = re.search(r"FINAL ANSWER:\s*(.*)", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
        else:
            answer = response.strip()
        
        return answer
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        return f"Error: {e}"


def test_simple_question():
    """Test with a simple question that doesn't require external APIs"""
    print("\n🧪 Testing with simple math question...")
    question = "What is 15 + 27?"
    answer = run_agent_cli(question)
    print(f"✅ Test result: {answer}")


def main():
    """Main CLI interface"""
    print("\n" + "="*60)
    print("🚀 AGENT COMMAND LINE INTERFACE")
    print("="*60)
    print("Welcome! You can ask questions and the agent will help you.")
    print("Type 'test' to run a simple test")
    print("Type 'quit', 'exit', or 'q' to stop the program.")
    print("="*60 + "\n")
    
    while True:
        try:
            # Get user input
            question = input("💭 Ask me anything (or 'test' for simple test): ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q', '']:
                print("\n👋 Goodbye! Thanks for using the agent.")
                break
            
            # Check for test command
            if question.lower() == 'test':
                test_simple_question()
                continue
            
            print("\n" + "-"*40)
            print("🤖 Agent is thinking...")
            print("-"*40)
            
            # Run the agent
            answer = run_agent_cli(question)
            
            print(f"\n✅ ANSWER:\n{answer}\n")
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the agent.")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()