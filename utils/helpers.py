from core.state import AgentState
import json
import os
import time

def log_agent_step(state: AgentState, step_name: str, status: str, **kwargs):
    """Safely records an agent step in state audit_log."""
    if isinstance(state.get("audit_log"), list):
        state["audit_log"].append({"step": step_name, "status": status, **kwargs})

def dump_agent_state(state: AgentState, agent_name: str, log_dir: str = "data/logs/state_dumps"):
    """Dumps AgentState to JSON. Off by default — writing under the repo restarts uvicorn --reload."""
    if os.getenv("DUMP_AGENT_STATE", "false").lower() not in ("1", "true", "yes"):
        return
    os.makedirs(log_dir, exist_ok=True)
    
    safe_state = {}
    for k, v in state.items():
        if k == "messages" and v:
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
