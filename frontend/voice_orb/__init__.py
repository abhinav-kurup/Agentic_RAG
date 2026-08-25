import os

import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(__file__)

_voice_orb = components.declare_component("documind_voice_orb", path=_COMPONENT_DIR)


def voice_orb(
    api_url: str,
    session_id: str,
    last_answer: str = "",
    pending_read: str = "",
    enable_tts: bool = True,
    key=None,
):
    """Floating talk orb. Returns a turn/confirm/error event when the iframe finishes work."""
    return _voice_orb(
        api_url=api_url or "",
        session_id=session_id or "",
        last_answer=last_answer or "",
        pending_read=pending_read or "",
        enable_tts=bool(enable_tts),
        key=key,
        default=None,
    )
