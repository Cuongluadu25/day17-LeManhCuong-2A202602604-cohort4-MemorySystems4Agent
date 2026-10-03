from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, is_question
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


_STOP_WORDS = {
    "mình", "tôi", "bạn", "của", "và", "những", "các", "là", "có", "không",
    "cho", "với", "này", "kia", "đó", "được", "như", "thế", "nào", "gì",
    "đang", "hiện", "vẫn", "còn", "nữa",
}


class BaselineAgent:
    """Agent A: only short-term (within-thread) memory.

    - Keeps messages per `thread_id`.
    - No persistent `User.md`, so it forgets facts as soon as a new thread
      starts (this is the fair comparison point against the advanced agent).
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        state = self.sessions.setdefault(thread_id, SessionState())

        # The whole history is re-read every turn: this is the naive behavior
        # that makes baseline prompt cost explode on long threads.
        context_tokens = sum(estimate_tokens(m["content"]) for m in state.messages) + estimate_tokens(message)
        state.prompt_tokens_processed += context_tokens

        state.messages.append({"role": "user", "content": message})
        reply_text = self._offline_response(thread_id, message)
        reply_tokens = estimate_tokens(reply_text)

        state.messages.append({"role": "assistant", "content": reply_text})
        state.token_usage += reply_tokens

        return {"reply": reply_text, "agent_tokens": reply_tokens, "prompt_tokens": context_tokens}

    def _offline_response(self, thread_id: str, message: str) -> str:
        state = self.sessions.get(thread_id)
        history = [m["content"] for m in (state.messages if state else [])]

        if is_question(message):
            # Naive short-term recall: only searches the *current* thread.
            keywords = [w for w in re.findall(r"\w{3,}", message) if w.lower() not in _STOP_WORDS]
            for past in reversed(history):
                if any(k.lower() in past.lower() for k in keywords):
                    return "Trong cuộc trò chuyện này, tôi nhớ bạn từng nói: " + past[:160]
            return "Tôi chưa có thông tin đó trong cuộc trò chuyện này."
        return "Đã ghi nhận."

    def _maybe_build_langchain_agent(self) -> None:
        if self.force_offline:
            self.langchain_agent = None
            return
        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        try:
            from langchain_core.messages import HumanMessage

            response = self.langchain_agent.invoke([HumanMessage(content=message)])
            text = response.content if hasattr(response, "content") else str(response)
            reply_tokens = estimate_tokens(text)
            state = self.sessions.setdefault(thread_id, SessionState())
            state.token_usage += reply_tokens
            return {"reply": text, "agent_tokens": reply_tokens, "prompt_tokens": estimate_tokens(message)}
        except Exception:
            return self._reply_offline(thread_id, message)
