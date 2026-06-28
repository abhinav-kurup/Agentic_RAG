import logging
from langsmith import traceable
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.ollama import OllamaModel
from core.llm import get_llm
from core.state import AgentState
from core.config import Config
from utils.helpers import log_agent_step, dump_agent_state

logger = logging.getLogger(__name__)


class AnalysisResult(BaseModel):
    answer: str = Field(description="The final answer to the user query based on the context.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    citations: List[str] = Field(default_factory=list, description="Citations format: [Source X (Page Y)].")


def get_pydantic_ai_model(model_identifier: str):
    if model_identifier.startswith("gemini/"):
        model_name = model_identifier.split("gemini/")[1]
        print("mmmmmmmmmmmmmmmmmmmmmmm",model_name)

        return GeminiModel(model_name)
    elif model_identifier.startswith("groq/"):
        model_name = model_identifier.split("groq/")[1]
        print("qqqqqqqqqqqqqqqqqq", model_name)

        return GroqModel(model_name)
    elif model_identifier.startswith("ollama/"):
        model_name = model_identifier.split("ollama/")[1]
        print("ooooooooooooooooooooo", model_name)
        return OllamaModel(model_name, base_url=Config.OLLAMA_BASE_URL)
    else:
        print("lllllllllllllllllllllllllll", model_identifier)
        return OllamaModel(model_identifier, base_url=Config.OLLAMA_BASE_URL)


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for all math operations.
    
    Args:
        expression: The math expression to evaluate, e.g. '2 + 2' or 'math.sqrt(16)'.
    """
    try:
        logger.info(f"ToolCalling for expression: {expression}")
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        import math
        allowed_names.update({k: v for k, v in math.__dict__.items() if not k.startswith("__")})
        return str(eval(expression, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"Error evaluating expression: {e}"

class AnalysisAgent:
    def __init__(self):
        model_identifier = Config.ANALYSIS_MODEL
        logger.info(f"AnalysisAgent initialized with Model: {model_identifier}")
        
        self.model = get_pydantic_ai_model(model_identifier)
        from agents.prompt import ANALYSIS_PYDANTIC_AI_SYSTEM_PROMPT
        self.agent = Agent(
            self.model,
            output_type=AnalysisResult,
            system_prompt=ANALYSIS_PYDANTIC_AI_SYSTEM_PROMPT.format()
        )
        self.agent.tool_plain(calculator)

    @traceable(name="Analysis")
    def invoke(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "AnalysisAgent")
        logger.info("AnalysisAgent: Process started")
        
        query = state.get("query", "")
        docs = state.get("retrieved_docs", [])
        extracted = state.get("extracted_data", {}).get("content", "")
        
        context_parts = []
        for i, doc in enumerate(docs):
            filename = doc['metadata'].get('source', f'Document {i+1}')
            page = doc['metadata'].get('page_number', '?')
            source = f"{filename} (Page {page})"
            context_parts.append(f"[{source}]: {doc.get('content', '')}")
            
        context_str = "\n\n".join(context_parts)
        if extracted:
            context_str += f"\n\n[Extracted Data]:\n{extracted}"

        user_prompt = f"Context:\n{context_str}\n\nQuery: {query}"

        try:
            logger.info("AnalysisAgent: Invoking PydanticAI Agent...")
            response = self.agent.run_sync(user_prompt)
            print("RRRRRRRRRRR",response)
            
            result: AnalysisResult = response.output
            logger.info("AnalysisAgent: Generation successful")
            
            log_agent_step(state, "AnalysisAgent", "Success")
            
            return {
                "final_response": result.answer,
                "citations": result.citations,
                "audit_log": [{
                    "step": "AnalysisAgent", 
                    "status": "Success", 
                    "confidence": result.confidence
                }]
            }
            
        except Exception as e:
            err = repr(e)
            logger.exception("AnalysisAgent generation failed: %s", err)
            log_agent_step(state, "AnalysisAgent", "Error", error=err, phase="invoke")
            return {
                "final_response": (
                    "I apologize, but I encountered a system issue while analyzing "
                    "the documents. Please check the Audit Logs or verify the AI "
                    "model is correctly configured."
                ),
                "audit_log": [{
                    "step": "AnalysisAgent",
                    "status": "Error",
                    "error": err,
                    "phase": "invoke",
                }],
            }
