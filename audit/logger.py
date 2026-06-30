import json
import os
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class AuditLogger:
    def __init__(self, log_dir: str = "data/logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "audit_trail.jsonl")

    def log_step(self, query_id: str, step_name: str, status: str, **kwargs):
        entry = {
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(),
            "step": step_name,
            "status": status,
            **kwargs
        }
        
        try:
            step_log_file = self.log_file.replace(".jsonl", "_steps.jsonl")
            with open(step_log_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(f"Logged step: {step_name} - {status}")
        except Exception as e:
            logger.error(f"Failed to write step log: {e}")
    
    def log_query(self, query_id: str, state: Dict[str, Any]):
        entry = {
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(),
            "query": state.get("query"),
            "final_response": state.get("final_response"),
            "audit_trail": state.get("audit_log", []),
            "retrieved_docs": state.get("retrieved_docs", [])
        }
        
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def get_logs(self) -> List[Dict[str, Any]]:
        logs = []
        if not os.path.exists(self.log_file):
            return logs

        try:
            with open(self.log_file, "r", encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read audit logs: {e}")
        
        return logs
