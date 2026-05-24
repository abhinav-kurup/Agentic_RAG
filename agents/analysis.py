from core.state import AgentState
from core.config import Config
from typing import Dict, Any
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from utils.helpers import log_agent_step, dump_agent_state
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for all math operations."""
    try:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        import math
        allowed_names.update({k: v for k, v in math.__dict__.items() if not k.startswith("__")})
        return str(eval(expression, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"Error evaluating expression: {e}"

class AnalysisAgent:
    def __init__(self, model_identifier: str = None):
        from core.llm import get_llm
        model_identifier = model_identifier or Config.MODEL_NAME
        logger.info(f"AnalysisAgent initialized with Model: {model_identifier}")
        self.llm = get_llm(model_identifier, temperature=0.2)
        self.tools = [calculator]

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "AnalysisAgent")

        logger.info("AnalysisAgent: Process started")
        query = state.get("query", "")
        docs = state.get("retrieved_docs", [])
        extracted = state.get("extracted_data", {}).get("content", "")
        messages = state.get("messages", [])
        
        context_parts = []
        for i, doc in enumerate(docs):
            filename = doc['metadata'].get('source', f'Document {i+1}')
            page = doc['metadata'].get('page_number', '?')
            source = f"{filename} (Page {page})"
            context_parts.append(f"[{source}]: {doc.get('content', '')}")
            
        context_str = "\n\n".join(context_parts)
        
        if extracted:
            context_str += f"\n\n[Extracted Data]:\n{extracted}"

        system_prompt = f"""You are an intelligent document analysis assistant. 
Answer the user's query based ONLY on the provided context.
If the answer is not in the context, state that you don't know.
You have access to tools. Use them if you need to perform calculations or operations you cannot do reliably yourself.

IMPORTANT RULES:
- If [Extracted Data] is provided in the context, it contains the exact, pre-processed information the user requested. 
- You MUST present this extracted data to the user directly and clearly.
- Do not be overly literal; if the user asks for a "table" and the extracted data is JSON, format that JSON into a beautifully formatted Markdown table.
- Do not claim information is missing if it is present in the [Extracted Data] section.

CITATION RULES:
- You must cite your sources using the format [Source X (Page Y)].
- NEVER cite "[Extracted Data]" as a source. If you are using information from the [Extracted Data] block, cite the original document name provided above it in the context.

Context:
{context_str}
"""

        llm_with_tools = self.llm.bind_tools(self.tools)
        
        try:
            logger.info("AnalysisAgent: Invoking LLM for generation...")
            if not messages:
                user_msg = HumanMessage(content=query)
                input_messages = [SystemMessage(content=system_prompt), user_msg]
                response = llm_with_tools.invoke(input_messages)
                new_messages = [user_msg, response]
            else:
                input_messages = [SystemMessage(content=system_prompt)] + messages
                response = llm_with_tools.invoke(input_messages)
                new_messages = [response]
            
            logger.info("AnalysisAgent: Generation successful")
            
            result_dict = {"messages": new_messages}
            
            if not response.tool_calls:
                result_dict["final_response"] = response.content
                log_agent_step(state, "AnalysisAgent", "Success", response_length=len(response.content))
                result_dict["audit_log"] = [{
                    "step": "AnalysisAgent", 
                    "status": "Success", 
                    "response_length": len(response.content)
                }]
            else:
                log_agent_step(state, "AnalysisAgent", "ToolCall", tool_calls=len(response.tool_calls))
                result_dict["audit_log"] = [{
                    "step": "AnalysisAgent", 
                    "status": "ToolCall", 
                    "tool_calls": [t["name"] for t in response.tool_calls]
                }]
            
            return result_dict
            
        except Exception as e:
            logger.error(f"AnalysisAgent Error: {e}")
            log_agent_step(state, "AnalysisAgent", "Error", error=str(e))
            return {
                "final_response": "I apologize, but I encountered a system issue while analyzing the documents. Please check the Audit Logs or verify the AI model is correctly configured.",
                "audit_log": [{
                    "step": "AnalysisAgent", 
                    "status": "Error", 
                    "error": str(e)
                }]
            }
