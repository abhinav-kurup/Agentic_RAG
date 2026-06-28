from core.state import AgentState
import json
import os
import time

def log_agent_step(state: AgentState, step_name: str, status: str, **kwargs):
    """Safely logs an agent step if the logger exists in state."""
    audit_logger = state.get("audit_logger")
    query_id = state.get("query_id")
    if audit_logger and query_id:
        audit_logger.log_step(query_id, step_name, status, **kwargs)

def dump_agent_state(state: AgentState, agent_name: str, log_dir: str = "data/logs/state_dumps"):
    """Dumps the full AgentState to a JSON file for debugging."""
    os.makedirs(log_dir, exist_ok=True)
    
    safe_state = {}
    for k, v in state.items():
        if k == "audit_logger":
            safe_state[k] = "<AuditLogger Object>"
        elif k == "messages" and v:
            safe_state[k] = [{"role": getattr(m, "type", "unknown"), "content": getattr(m, "content", "")} for m in v]
        else:
            safe_state[k] = v
            
    query_id = state.get("query_id", "unknown_query_id")
    timestamp = str(int(time.time() * 1000))
    filename = os.path.join(log_dir, f"{query_id}_{timestamp}_{agent_name}_state.json")
    
    def serialize_unknown(obj):
        if isinstance(obj, set):
            return sorted(list(obj))
        try:
            return str(obj)
        except Exception:
            return "<Unserializable Object>"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(safe_state, f, indent=2, ensure_ascii=False, default=serialize_unknown)
    except Exception as e:
        print(f"Failed to dump state for {agent_name}: {e}")
