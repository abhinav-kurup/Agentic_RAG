from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import os



def get_llm(model_identifier: str, temperature: float = 0.2, timeout: int = 30):
    """
    Returns a LangChain BaseChatModel based on the model_identifier.
    Examples format:
      - "gemini/gemini-2.5-pro"
      - "ollama/qwen2.5:3b"
      - "ollama/llama3"
    If no prefix is provided, defaults to ollama.
    """
    if model_identifier.startswith("gemini/"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = model_identifier.split("gemini/")[1]
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            timeout=timeout
        )
    elif model_identifier.startswith("ollama/"):
        model_name = model_identifier.split("ollama/")[1]
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout
        )
    elif model_identifier.startswith("groq/"):
        model_name = model_identifier.split("groq/")[1]
        return ChatGroq(
            model=model_name,
            temperature=temperature
        )
    else:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model_identifier,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout
        )


