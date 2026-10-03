from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    is_question,
)
from model_provider import build_chat_model


_FACT_LABELS = {
    "name": "tên",
    "location": "nơi ở",
    "profession": "nghề nghiệp",
    "style": "style trả lời",
    "drink": "đồ uống yêu thích",
    "food": "món ăn yêu thích",
    "pet": "thú cưng",
    "interests": "mối quan tâm",
}

_CATEGORY_TRIGGERS = {
    "name": (r"tên", r"là ai", r"gọi là gì"),
    "location": (r"ở đâu", r"nơi ở", r"đang ở", r"sống ở", r"\bở\b"),
    "profession": (r"nghề", r"làm gì", r"công việc", r"engineer"),
    "style": (r"style", r"trả lời", r"bullet", r"cách (?:trả lời|nói)"),
    "drink": (r"đồ uống", r"thức uống", r"\buống\b"),
    "food": (r"món ăn", r"\băn\b"),
    "pet": (r"nuôi", r"con gì", r"thú cưng"),
    "interests": (r"quan tâm", r"sở thích", r"mối quan tâm"),
}


def _merge_interests(old: str, new: str) -> str:
    """Union of two comma-separated interest lists, preserving first-seen order."""
    seen: list[str] = []
    for part in (old.split(",") + new.split(",")):
        token = part.strip()
        if token and token not in seen:
            seen.append(token)
    return ", ".join(seen)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term + persistent `User.md` + compact memory.

    Flow per turn:
      extract_profile_updates() -> User.md -> CompactMemoryManager.append()
      -> prompt = User.md + summary + recent messages -> reply -> update tokens.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None
        self._maybe_build_langchain_agent()

    # --- public API -------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # --- offline path -----------------------------------------------------

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1. Extract stable facts and persist them into User.md.
        #    Questions are skipped: they ask for info instead of providing it,
        #    so extracting from them would pollute the profile.
        if not is_question(message):
            updates = extract_profile_updates(message)
            existing = self.profile_store.facts(user_id)
            for key, value in updates.items():
                # Interests accumulate (a user keeps old + new tech interests).
                if key == "interests" and key in existing:
                    value = _merge_interests(existing[key], value)
                self.profile_store.upsert_fact(user_id, key, value)

        # 2. Append the user message to short-term memory (auto-compact).
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate the prompt context carried into this turn.
        context_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + context_tokens

        # 4. Generate the answer (recall questions are served from User.md).
        reply_text = self._offline_response(user_id, thread_id, message)

        # 5. Append the assistant reply and update token counters.
        self.compact_memory.append(thread_id, "assistant", reply_text)
        reply_tokens = estimate_tokens(reply_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens

        return {"reply": reply_text, "agent_tokens": reply_tokens, "prompt_tokens": context_tokens}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        total = estimate_tokens(self.profile_store.read_text(user_id))
        ctx = self.compact_memory.context(thread_id)
        total += estimate_tokens(ctx.get("summary") or "")
        for m in ctx.get("messages", []):
            total += estimate_tokens(m["content"])
        return total

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        if not is_question(message):
            return "Đã ghi nhớ thông tin của bạn."
        facts = self.profile_store.facts(user_id)
        return self._answer_from_facts(facts, message)

    def _answer_from_facts(self, facts: dict[str, str], question: str) -> str:
        categories = self._detect_categories(question)
        if not categories:
            categories = [k for k in _FACT_LABELS if k in facts]

        parts = []
        for cat in categories:
            if cat in facts:
                parts.append(f"{_FACT_LABELS[cat]} của bạn là {facts[cat]}")

        if not parts:
            return "Mình chưa lưu đủ thông tin về bạn trong hồ sơ để trả lời câu này."
        return "Dựa trên hồ sơ đã lưu: " + "; ".join(parts) + "."

    def _detect_categories(self, question: str) -> list[str]:
        q = question.lower()
        cats = []
        for cat, patterns in _CATEGORY_TRIGGERS.items():
            if any(re.search(p, q) for p in patterns):
                cats.append(cat)
        return cats

    # --- live path (optional extension) -----------------------------------

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
            from langchain_core.messages import HumanMessage, SystemMessage

            facts = self.profile_store.facts(user_id)
            system = "Bạn là trợ lý có trí nhớ dài hạn. Hồ sơ người dùng:\n" + "\n".join(
                f"- {k}: {v}" for k, v in sorted(facts.items())
            )
            response = self.langchain_agent.invoke(
                [SystemMessage(content=system), HumanMessage(content=message)]
            )
            text = response.content if hasattr(response, "content") else str(response)
            reply_tokens = estimate_tokens(text)
            self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens
            return {
                "reply": text,
                "agent_tokens": reply_tokens,
                "prompt_tokens": estimate_tokens(system + message),
            }
        except Exception:
            return self._reply_offline(user_id, thread_id, message)
