# services/gemini_client.py

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Initialize the LangChain LLM wrapper for Gemini
# This is the generic LLM instance that other services can use.
# You can configure temperature, top_p, etc., here if you want default settings.
try:
    # Use a generally available and performant model
    # Check langchain-google-genai documentation for the latest supported model names.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        # temperature=0.1 # Example: set a default low temperature for more factual responses
        # convert_system_message_to_human=True # May be needed for some models/tasks if system prompts are ignored
    )
except Exception as e:
    # Log this error, as it's a critical failure for services relying on the LLM
    print(f"FATAL: Could not initialize ChatGoogleGenerativeAI: {e}")
    # Depending on your application, you might want to raise it or handle it
    # such that the app can inform the user that AI services are unavailable.
    llm = None  # Or raise an exception to halt app startup if LLM is critical
    # raise ConnectionError(f"Failed to initialize Gemini LLM: {e}") from e

if __name__ == "__main__":
    if llm:
        print("Gemini LLM client initialized successfully.")
        try:
            from langchain_core.messages import HumanMessage

            response = llm.invoke([HumanMessage(content="Say hi!")])
            print(f"Test invocation: {response.content}")
        except Exception as e:
            print(f"Error during test invocation: {e}")
    else:
        print("Gemini LLM client failed to initialize.")
