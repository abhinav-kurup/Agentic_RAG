import asyncio
import logging
from langsmith import traceable
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from langchain_core.messages import AIMessage
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
        return GoogleModel(model_name)
    elif model_identifier.startswith("groq/"):
        model_name = model_identifier.split("groq/")[1]
        return GroqModel(model_name)
    elif model_identifier.startswith("ollama/"):
        model_name = model_identifier.split("ollama/")[1]
        return OllamaModel(
            model_name,
            provider=OllamaProvider(base_url=Config.OLLAMA_BASE_URL),
        )
    else:
        return OllamaModel(
            model_identifier,
            provider=OllamaProvider(base_url=Config.OLLAMA_BASE_URL),
        )


@traceable(name="CalculatorTool")
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
    """Synthesizes structured final answers with citations asynchronously using PydanticAI."""

    def __init__(self):
        model_identifier = Config.ANALYSIS_MODEL
        logger.info(f"AnalysisAgent initialized with Model: {model_identifier}")
        
        self.model = get_pydantic_ai_model(model_identifier)
        from agents.prompt import ANALYSIS_PYDANTIC_AI_SYSTEM_PROMPT
        self.agent = Agent(
            self.model,
            output_type=AnalysisResult,
            system_prompt=ANALYSIS_PYDANTIC_AI_SYSTEM_PROMPT.format(),
            retries=3,
        )
        self.agent.tool_plain(calculator)

    @traceable(name="Analysis")
    async def ainvoke(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "AnalysisAgent")
        logger.info("AnalysisAgent: Process started")
        
        query = state.get("standalone_query") or state.get("query", "")
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
            logger.info("AnalysisAgent: Invoking PydanticAI Agent asynchronously...")
            response = await self.agent.run(user_prompt)

            
            result: AnalysisResult = response.output
            logger.info(f"AnalysisAgent: Generation successful (Confidence: {result.confidence:.2f})")
            log_agent_step(state, "AnalysisAgent", "Success")
            
            return {
                "final_response": result.answer,
                "messages": [AIMessage(content=result.answer)],
                "citations": result.citations,
                "audit_log": [{
                    "step": "AnalysisAgent", 
                    "status": "Success", 
                    "confidence": result.confidence
                }]
            }
            
        except Exception as e:
            err = repr(e)
            logger.warning("AnalysisAgent primary PydanticAI run failed (%s). Executing direct LLM fallback...", err)
            try:
                fallback_llm = get_llm(Config.ANALYSIS_MODEL, temperature=0.1)
                fallback_resp = await fallback_llm.ainvoke(
                    f"Based on the following context, answer the user query clearly and accurately:\n\n{user_prompt}"
                )
                answer_text = getattr(fallback_resp, "content", str(fallback_resp)).strip()
                log_agent_step(state, "AnalysisAgent", "Success", method="fallback")
                return {
                    "final_response": answer_text,
                    "messages": [AIMessage(content=answer_text)],
                    "citations": [],
                    "audit_log": [{
                        "step": "AnalysisAgent",
                        "status": "Success",
                        "method": "fallback"
                    }]
                }
            except Exception as fallback_err:
                logger.exception("AnalysisAgent fallback generation also failed: %s", fallback_err)
                return {
                    "final_response": (
                        "I apologize, but I encountered an issue synthesizing the final response. "
                        "Please verify your query or try rephrasing."
                    ),
                    "audit_log": [{
                        "step": "AnalysisAgent",
                        "status": "Error",
                        "error": repr(fallback_err),
                        "phase": "invoke",
                    }],
                }


    def invoke(self, state: AgentState) -> Dict[str, Any]:
        return asyncio.run(self.ainvoke(state))
