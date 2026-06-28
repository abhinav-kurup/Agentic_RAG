from langsmith import traceable
from core.state import AgentState
from core.config import Config
from typing import Dict, Any
import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.helpers import log_agent_step
logger = logging.getLogger(__name__)

class ExtractionAgent:
    def __init__(self):
        from core.llm import get_llm
        model_identifier = Config.EXTRACTION_MODEL
        self.llm = get_llm(model_identifier, temperature=0.1)

    @traceable(name="Extraction")
    def invoke(self, state: AgentState) -> Dict[str, Any]:
        from utils.helpers import dump_agent_state
        dump_agent_state(state, "ExtractionAgent")

        logger.info("ExtractionAgent: Process started")
        query = state.get("query", "")
        docs = state.get("retrieved_docs", [])
        
        if not docs:
            self._log_skip(state, "No docs found")
            return {"audit_log": [{"step": "ExtractionAgent", "status": "Skipped", "reason": "No docs found"}]}

        if "extract" not in query.lower() and "table" not in query.lower():
            self._log_skip(state, "Query does not request extraction")
            return {"audit_log": [{"step": "ExtractionAgent", "status": "Skipped", "reason": "Query does not request extraction"}]}

        context = "\n\n".join([d.get("content", "") for d in docs])
        
        from agents.prompt import EXTRACTION_PROMPT
        prompt = EXTRACTION_PROMPT
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({"query": query, "context": context})
            
            log_agent_step(state, "ExtractionAgent", "Success", extracted_length=len(result))
            
            return {
                "extracted_data": {"content": result},
                "audit_log": [{
                    "step": "ExtractionAgent", 
                    "status": "Success", 
                    "extracted_length": len(result)
                }]
            }
        except Exception as e:
            logger.error(f"ExtractionAgent Error: {e}")
            log_agent_step(state, "ExtractionAgent", "Error", error=str(e))
            return {
                "audit_log": [{
                    "step": "ExtractionAgent", 
                    "status": "Error", 
                    "error": str(e)
                }]
            }
    
    def _log_skip(self, state: AgentState, reason: str):
        log_agent_step(state, "ExtractionAgent", "Skipped", reason=reason)
