import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_SPEAK = {
    "read it",
    "read that",
    "read this",
    "read aloud",
    "read it aloud",
    "read that aloud",
    "read the answer",
    "read the reply",
    "read it again",
    "read that again",
    "speak that",
    "speak it",
    "speak the answer",
    "say it out loud",
    "say that out loud",
    "say it aloud",
    "say it again",
    "say that again",
    "repeat",
    "repeat it",
    "repeat that",
    "repeat again",
    "repeat the answer",
    "repeat the last answer",
    "repeat last answer",
    "can you repeat",
    "can you repeat that",
    "can you repeat it",
    "can you repeat the answer",
    "can you repeat the last answer",
    "could you repeat",
    "could you repeat that",
    "could you repeat the last answer",
    "please repeat",
    "please repeat that",
    "please repeat the last answer",
}

_SPEAK_PREFIXES = (
    "repeat the last",
    "repeat last",
    "can you repeat",
    "could you repeat",
    "please repeat",
    "say that again",
    "say it again",
    "read it again",
    "read that again",
)

_QUIET = {
    "stop",
    "stop it",
    "stop now",
    "stop reading",
    "stop speaking",
    "stop talking",
    "don't read",
    "do not read",
    "dont read",
    "don't speak",
    "do not speak",
    "be quiet",
    "silence",
    "enough",
    "that's enough",
    "thats enough",
    "that is enough",
    "that's it",
    "thats it",
    "that is it",
    "never mind",
    "nevermind",
    "cut it",
    "cancel",
}

_STOP_PREFIXES = (
    "that's enough",
    "thats enough",
    "that is enough",
    "that's it",
    "stop reading",
    "stop speaking",
    "stop talking",
)

_HANGUP = {
    "goodbye",
    "good bye",
    "bye",
    "bye bye",
    "stop listening",
    "hang up",
    "end session",
    "that's all",
    "thats all",
}

_INCOMPLETE = {
    "what",
    "what is",
    "what are",
    "what was",
    "what were",
    "what about",
    "how",
    "how do",
    "how does",
    "how did",
    "how is",
    "why",
    "why is",
    "why are",
    "who",
    "who is",
    "when",
    "where",
    "tell me",
    "tell",
    "can you",
    "could you",
    "would you",
    "i want",
    "i need",
    "please",
    "um",
    "uh",
    "so",
    "and",
    "the",
    "a",
}

_YES = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "sure",
    "ok",
    "okay",
    "go ahead",
    "yes please",
    "yeah please",
    "yes read it",
    "read it",
    "read it aloud",
}

_NO = {
    "no",
    "nope",
    "nah",
    "no thanks",
    "no thank you",
    "don't",
    "dont",
    "skip",
    "not now",
}


def _normalize(transcript: str) -> str:
    text = (transcript or "").lower().strip()
    text = re.sub(r"^[.\s]+|[.\s!?]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bu\b", "you", text)
    text = re.sub(r"\bur\b", "your", text)
    text = re.sub(r"\bya\b", "yeah", text)
    text = text.replace("wanna", "want to").replace("gonna", "going to")
    if text.startswith("please "):
        text = text[7:].strip()
    return text


def _is_stop(text: str) -> bool:
    if text in _QUIET:
        return True
    if any(text.startswith(prefix) for prefix in _STOP_PREFIXES):
        return True
    return any(
        phrase in text
        for phrase in (
            "that's enough",
            "thats enough",
            "that is enough",
            "stop reading",
            "stop speaking",
            "stop talking",
        )
    )


def is_incomplete(transcript: str) -> bool:
    text = _normalize(transcript)
    if not text:
        return True
    if text in _INCOMPLETE:
        return True
    words = text.split()
    if len(words) <= 2 and words[0] in {
        "what",
        "how",
        "why",
        "who",
        "when",
        "where",
        "tell",
        "can",
        "could",
        "would",
    }:
        return True
    return False


def _is_speak(text: str) -> bool:
    if text in _SPEAK:
        return True
    return any(text.startswith(prefix) for prefix in _SPEAK_PREFIXES)


def classify_intent(transcript: str, awaiting_confirm: bool = False) -> str:
    """Return ask | speak | quiet | clarify | hangup | confirm_yes | confirm_no | unheard."""
    text = _normalize(transcript)
    if not text:
        return "unheard"
    if text in _HANGUP:
        return "hangup"
    if _is_stop(text):
        return "quiet"
    if _is_speak(text):
        return "speak"
    if awaiting_confirm and len(text) < 40:
        if text in _NO:
            return "confirm_no"
        if text in _YES:
            return "confirm_yes"
    if is_incomplete(text):
        return "clarify"
    return "ask"


def classify_voice_command(transcript: str) -> Optional[str]:
    intent = classify_intent(transcript, awaiting_confirm=False)
    if intent in ("speak", "quiet", "hangup"):
        return intent
    return None


_VALID = {"ask", "speak", "quiet", "clarify", "hangup", "unheard"}
_ROUTER_PROMPT = """You route a voice utterance for a document Q&A assistant.
Do not answer the document question. Return JSON only.

Intents:
- ask: they want information or a follow-up. Copy what they asked into "query". Do not merge in the previous topic.
- speak: they want the last answer read aloud (repeat, say that again, read it).
- quiet: stop talking but keep listening (stop, that's enough).
- hangup: end the voice session (goodbye, stop listening).
- clarify: ONLY if they said a fragment like "what is" or "um" with no real request.
- unheard: empty/noise.

When unsure, use ask, not clarify.
If they want the previous answer spoken again, use speak.

Chat history:
{history}

Previous answer available: {has_answer}

User said: {transcript}

JSON: {{"intent": "...", "query": ""}}
"""


class VoiceRouter:
    def __init__(self):
        from core.config import Config
        from core.llm import get_llm

        self.llm = get_llm(Config.VOICE_ROUTER_MODEL, temperature=0)

    async def aroute(
        self,
        transcript: str,
        history_text: str = "",
        has_last_answer: bool = False,
    ) -> dict:
        text = _normalize(transcript)
        if not text:
            return {"intent": "unheard", "query": ""}

        fast = classify_intent(transcript)
        if fast in ("speak", "quiet", "hangup", "unheard"):
            return {"intent": fast, "query": ""}

        prompt = _ROUTER_PROMPT.format(
            history=history_text or "(none)",
            has_answer="yes" if has_last_answer else "no",
            transcript=(transcript or "").strip(),
        )
        try:
            from langchain_core.messages import HumanMessage

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            parsed = _parse_router_output(getattr(response, "content", "") or "")
            if parsed:
                if parsed["intent"] == "clarify" and len(text.split()) >= 3:
                    parsed["intent"] = "ask"
                    if not parsed.get("query"):
                        parsed["query"] = (transcript or "").strip()
                return parsed
        except Exception:
            logger.exception("Voice router LLM failed; using phrase fallback")
        return {"intent": fast if fast != "unheard" else "ask", "query": ""}


def _parse_router_output(raw: str) -> Optional[dict]:
    blob = (raw or "").strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```(?:json)?\s*|\s*```$", "", blob).strip()
    match = re.search(r"\{.*\}", blob, re.DOTALL)
    if match:
        blob = match.group(0)
    try:
        data = json.loads(blob)
    except Exception:
        return None
    intent = str(data.get("intent") or "").lower().strip()
    if intent in ("confirm_yes", "read", "repeat"):
        intent = "speak"
    if intent in ("confirm_no", "stop"):
        intent = "quiet"
    if intent not in _VALID:
        return None
    query = str(data.get("query") or "").strip()
    if intent != "ask":
        query = ""
    return {"intent": intent, "query": query}


_router = None


def get_voice_router() -> VoiceRouter:
    global _router
    if _router is None:
        _router = VoiceRouter()
    return _router
