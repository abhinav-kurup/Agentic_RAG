from core.state import AgentState
from core.config import Config
from typing import Dict, Any
import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.helpers import log_agent_step

logger = logging.getLogger(__name__)

class RouterAgent:
    """
    Routes queries using a micro LLM classifier to determine if they need 
    document retrieval or should be rejected as out-of-scope.
    """
    
    def __init__(self, model_identifier: str = None):
        from core.llm import get_llm
        model_identifier = model_identifier or Config.MODEL_NAME
        self.llm = get_llm(model_identifier, temperature=0.0)
        

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        """
        Classifies query using LLM into: 'document' or 'conversational'
        """
        from utils.helpers import dump_agent_state
        dump_agent_state(state, "RouterAgent")

        query = state.get("query", "").strip()
        
        # LLM Classification
        prompt = ChatPromptTemplate.from_template(
            """You are a query classifier for a document analysis system.

                Classify the user's query into ONE of these categories:

                1. "document" - Questions about specific information, data, or analysis that would require searching through documents
                Examples: "What is the revenue?", "Summarize page 5", "Extract sales data"

                2. "conversational" - Greetings, pleasantries, or questions about the system itself (not about documents)
                Examples: "Hello", "How are you?", "What can you do?", "Thanks"

                Query: "{query}"

                Respond with ONLY one word: "document" or "conversational"
            """
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            logger.info(f"RouterAgent: Classifying query with LLM...")
            classification = chain.invoke({"query": query}).strip().lower()
            
            # Parse LLM response
            if "document" in classification:
                route = "document"
            elif "conversational" in classification:
                route = "conversational"
            else:
                logger.warning(f"Unclear classification '{classification}', defaulting to 'document'")
                route = "document"
            
            logger.info(f"RouterAgent: LLM classified as '{route}'")
            

            log_agent_step(
                state=state,
                step_name="RouterAgent",
                status="Success",
                route=route,
                method="llm",
                classification=classification
            )
            
            return {
                "route": route,
                "audit_log": [{
                    "step": "RouterAgent",
                    "status": "Success",
                    "route": route,
                    "method": "llm",
                    "classification": classification
                }]
            }
            
        except Exception as e:
            logger.error(f"RouterAgent LLM Error: {e}") 
            route = self._fallback_classification(query)
            
            logger.info(f"RouterAgent: Fallback classified as '{route}'")
            

            log_agent_step(
                state=state,
                step_name="RouterAgent",
                status="Success",
                route=route,
                method="fallback",
                error=str(e)
            )
            
            return {
                "route": route,
                "audit_log": [{
                    "step": "RouterAgent",
                    "status": "Success",
                    "route": route,
                    "method": "fallback",
                    "error": str(e)
                }]
            }
    
    def _fallback_classification(self, query: str) -> str:
        query_lower = query.lower()
        
        greetings = ['hello', 'hi', 'hey', 'thanks', 'thank', 'bye']
        for greeting in greetings:
            if greeting in query_lower:
                return "conversational"
        
        if len(query.split()) < 3 and '?' in query:
            return "conversational"
        
        return "document"

def reject_query(state):
    from utils.helpers import dump_agent_state
    dump_agent_state(state, "RejectionAgent")

    response = (
        "I'm sorry, but I can only answer questions about the documents you've uploaded. "
        "Your query appears to be conversational or out of scope.\n\n"
        "Please ask me questions about your documents, such as:\n"
        "- What is the revenue mentioned in the report?\n"
        "- Summarize the key findings\n"
        "- Extract data from tables"
    )
    
    log_agent_step(state, "RejectionHandler", "Out-of-Scope", query=state.get("query"))
    
    return {
        "final_response": response,
        "audit_log": [{"step": "RejectionHandler", "status": "Out-of-Scope"}]
    }

def no_docs_found_reply(state):
    from utils.helpers import dump_agent_state
    dump_agent_state(state, "NoDocsFoundAgent")

    return {
        "final_response": "I'm sorry, but I can't find any documents that match your query.",
        "audit_log": [{"step": "NoDocsFoundAgent", "status": "NoDocsFound"}]
    }