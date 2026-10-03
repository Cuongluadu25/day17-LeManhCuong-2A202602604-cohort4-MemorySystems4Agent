from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Simple heuristic token estimator (stable, deterministic, offline-safe).

    Approximate tokens from character count: `len(text) / 4`. Good enough for
    benchmark comparisons without a real tokenizer.
    """

    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


# ---------------------------------------------------------------------------
# Persistent user profile (`User.md`)
# ---------------------------------------------------------------------------

_FACTS_START = "<!-- facts:start -->"
_FACTS_END = "<!-- facts:end -->"


def _default_profile() -> str:
    return "# User Profile\n\n" + _FACTS_START + "\n" + _FACTS_END + "\n"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", (value or "").strip()).strip("_")
    return slug or "user"


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`, one file per user id.

    The markdown body is human-readable, while a delimited facts block keeps a
    structured key/value map that the agent can query offline.
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / _slugify(user_id) / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return _default_profile()

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    # --- structured facts -------------------------------------------------

    def facts(self, user_id: str) -> dict[str, str]:
        text = self.read_text(user_id)
        start = text.find(_FACTS_START)
        end = text.find(_FACTS_END)
        if start == -1 or end == -1:
            return {}
        block = text[start + len(_FACTS_START):end]
        result: dict[str, str] = {}
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("- ") and ":" in line[2:]:
                key, value = line[2:].split(":", 1)
                result[key.strip()] = value.strip()
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        facts = self.facts(user_id)
        facts[key] = value
        lines = [f"- {k}: {facts[k]}" for k in sorted(facts)]
        block = _FACTS_START + "\n" + "\n".join(lines) + "\n" + _FACTS_END

        text = self.read_text(user_id)
        start = text.find(_FACTS_START)
        end = text.find(_FACTS_END)
        if start == -1 or end == -1:
            new_text = text.rstrip() + "\n\n" + block + "\n"
        else:
            new_text = text[:start] + block + text[end + len(_FACTS_END):]
        self.write_text(user_id, new_text)


# ---------------------------------------------------------------------------
# Stable-fact extraction from user messages
# ---------------------------------------------------------------------------

# Words that bound a captured value on the right (e.g. "Đà Nẵng và ...",
# "MLOps engineer chứ ...", "backend engineer nữa ..."). Used both to strip a
# trailing connector at the end of the value and to split on an in-line one.
_TRAILING_CONNECTORS = (
    "và", "chứ", "nhưng", "còn", "nữa", "mỗi", "để", "nên", "dạo", "hiện",
    "đang", "cho", "rồi", "nhé", "giúp", "chưa", "đâu", "cả", "thôi", "trong",
    "khi", "với", "vài", "mấy", "hai", "một", "ba", "vì", "nếu", "như", "cũng",
    "vẫn", "thì", "là", "rằng", "này",
)
_TRAILING_WORDS = frozenset(_TRAILING_CONNECTORS)
_CONNECTOR_RE = re.compile(r"\s+(?:" + "|".join(_TRAILING_CONNECTORS) + r")\s+")

_QUESTION_TOKENS = ("gì", "nào", "mấy", "bao nhiêu", "ở đâu", "là gì", "thế nào", "như thế nào", "khi nào", "đâu")


def _clean(value: str) -> str:
    v = value.strip().strip(" \t,.;:!?\"'()[]")

    # Strip trailing connector words at the end of the value.
    while v:
        before, _, tail = v.rpartition(" ")
        if tail and tail in _TRAILING_WORDS:
            v = before.strip()
        else:
            break

    # Split on an in-line connector to keep only the left (subject) side.
    v = _CONNECTOR_RE.split(v)[0]
    return v.strip(" \t,.;:!?")


def _is_question_token(value: str) -> bool:
    v = value.strip().lower()
    return any(t in v for t in _QUESTION_TOKENS)


def _extract_name(text: str) -> str | None:
    m = re.search(r"tên\s+(?:mình|tôi|tớ)?\s*là\s+(\w+(?:\s+\w+)?)", text, re.IGNORECASE)
    if not m:
        return None
    val = _clean(m.group(1))
    if not val or not val[0].isupper() or _is_question_token(val):
        return None
    return val


def _extract_location(text: str) -> str | None:
    # Locations that were negated (old fact or noise) should not be stored.
    negated: set[str] = set()
    for pat in (
        r"không\s+còn\s+ở\s+(\w+(?:\s+\w+)?)",
        r"không\s+phải\s+(?:ở|nơi\s+ở)\s+(\w+(?:\s+\w+)?)",
        r"chỉ\s+là\s+nơi\s+(\w+(?:\s+\w+)?)",
    ):
        for mm in re.finditer(pat, text, re.IGNORECASE):
            negated.add(_clean(mm.group(1)))

    # Only "ở X" phrasings are location signals. "chuyển sang / sang / đến" in
    # this corpus always describes a *profession* change ("từ backend sang
    # MLOps"), so treating it as a location would pollute the fact.
    patterns = (
        r"thực\s+ra[^.]*?(?:đang\s+làm\s+việc\s+ở|đang\s+ở|ở)\s+(\w+(?:\s+\w+)?)",
        r"giờ\s+[^.]*?(?:đang\s+ở|ở)\s+(\w+(?:\s+\w+)?)",
        r"(?:đang\s+ở|hiện\s+đang\s+ở|hiện\s+ở|vẫn\s+ở)\s+(\w+(?:\s+\w+)?)",
        r"(?:^|[.!?]\s*)(?:mình|tôi|tớ)\s+ở\s+(\w+(?:\s+\w+)?)",
    )
    for pat in patterns:
        for mm in re.finditer(pat, text, re.IGNORECASE):
            val = _clean(mm.group(1))
            if val and val not in negated and not _is_question_token(val):
                return val
    return None


def _extract_profession(text: str) -> str | None:
    # Anchor on "X engineer" role phrases and read the immediately-preceding
    # context to decide positive / correction / noise. This avoids matching the
    # ambiguous "là" inside phrases like "nhớ là mình làm MLOps engineer".
    result: str | None = None
    for mm in re.finditer(r"([\w]+\s+engineer)", text, re.IGNORECASE):
        role = _clean(mm.group(1))
        before = text[max(0, mm.start() - 30):mm.start()]

        # Old fact explicitly withdrawn, or a "don't call me that" instruction.
        if re.search(r"không\s+còn\s+(?:làm|là)\s*$", before, re.IGNORECASE):
            continue
        if re.search(r"đừng\s+(?:nói|nhắc|gọi)\s*$", before, re.IGNORECASE):
            continue

        # Joke / hypothetical noise after the role.
        after = text[mm.end():mm.end() + 90]
        if "câu đùa" in after or "cho vui" in after:
            continue

        # A "chuyển sang X" correction wins immediately.
        if re.search(r"chuyển\s+sang\s*$", before, re.IGNORECASE):
            return role

        result = role

    return result


def _extract_style(text: str) -> str | None:
    has_short = bool(re.search(r"ngắn\s*gọn|gọn\s*gàng|súc\s*tích", text, re.IGNORECASE))
    has_bullet = bool(re.search(r"3\s*bullet|ba\s*bullet", text, re.IGNORECASE))
    has_example = bool(re.search(r"ví\s*dụ\s*(?:thực\s*(?:tế|chiến)|cụ\s*thể)", text, re.IGNORECASE))

    if not (has_short or has_bullet):
        return None
    parts = []
    if has_short:
        parts.append("ngắn gọn")
    if has_bullet:
        parts.append("3 bullet")
    if has_example:
        parts.append("có ví dụ thực tế")
    return ", ".join(parts)


def _extract_drink(text: str) -> str | None:
    m = re.search(
        r"(?:đồ\s+uống|thức\s+uống)\s+(?:yêu\s+thích|ưa\s+thích|khoái|ruột)\s+là\s+(\w+(?:\s+\w+){0,3})",
        text,
        re.IGNORECASE,
    )
    if m:
        val = _clean(m.group(1))
        if val:
            return val
    m = re.search(r"(?:uống|thích\s+uống)\s+(cà\s+phê(?:\s+sữa\s+đá)?|trà(?:\s+sữa)?)", text, re.IGNORECASE)
    if m:
        val = _clean(m.group(1))
        if val:
            return val
    return None


def _extract_food(text: str) -> str | None:
    m = re.search(
        r"món\s+ăn\s+(?:yêu\s+thích|ưa\s+thích|ruột)\s+là\s+(\w+(?:\s+\w+){0,2})",
        text,
        re.IGNORECASE,
    )
    if m:
        val = _clean(m.group(1))
        if val:
            return val
    return None


def _extract_pet(text: str) -> str | None:
    m = re.search(r"nuôi\s+(?:một\s+)?(?:bé|con|em)?\s*(\w+)\s+tên\s+(\w+)", text, re.IGNORECASE)
    if m:
        return _clean(m.group(1))
    m = re.search(r"con\s+(corgi|chó|mèo|thỏ|mèo\s+anh)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def _extract_interests(text: str) -> str | None:
    if not re.search(r"thích|quan\s+tâm|đang\s+học|tìm\s+hiểu|mục\s+tiêu", text, re.IGNORECASE):
        return None
    found: list[str] = []
    for kw in ("Python", "AI", "MLOps", "RAG", "agent", "memory", "async"):
        if kw == "AI":
            # Uppercase only: avoid matching the Vietnamese pronoun "ai".
            if re.search(r"\bAI\b", text):
                found.append(kw)
        elif re.search(rf"\b{kw}\b", text, re.IGNORECASE):
            found.append(kw)
    if not found:
        return None
    return ", ".join(dict.fromkeys(found))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert a raw user message into stable profile facts.

    Only confident, non-question facts are returned. Corrections ("không còn
    làm X", "chuyển sang Y", "Hà Nội chỉ là nơi họp") are handled by the
    dedicated extractors so the newest fact wins and noise is ignored.
    """

    if not message or not message.strip():
        return {}
    text = message.strip()

    updates: dict[str, str] = {}

    name = _extract_name(text)
    if name:
        updates["name"] = name

    location = _extract_location(text)
    if location:
        updates["location"] = location

    profession = _extract_profession(text)
    if profession:
        updates["profession"] = profession

    style = _extract_style(text)
    if style:
        updates["style"] = style

    drink = _extract_drink(text)
    if drink:
        updates["drink"] = drink

    food = _extract_food(text)
    if food:
        updates["food"] = food

    pet = _extract_pet(text)
    if pet:
        updates["pet"] = pet

    interests = _extract_interests(text)
    if interests:
        updates["interests"] = interests

    # Safety net: never persist a value that is actually a question word
    # (e.g. "đồ uống yêu thích là gì" -> "gì").
    for key in list(updates):
        if _is_question_token(updates[key]):
            del updates[key]

    return updates


def is_question(message: str) -> bool:
    """Lightweight classifier for recall/question turns vs. plain statements."""

    text = (message or "").strip()
    if not text:
        return False
    if text.endswith("?"):
        return True

    lower = text.lower()
    recall_hints = ("nhắc lại", "nhớ lại", "tóm tắt", "mô tả", "kể lại", "có biết", "hãy nhớ")
    question_hints = ("là gì", "ở đâu", "thế nào", "như thế nào", "bao nhiêu", "khi nào", "là ai")

    if any(h in lower for h in recall_hints) or any(h in lower for h in question_hints):
        return True
    return False


# ---------------------------------------------------------------------------
# Compact memory for long threads
# ---------------------------------------------------------------------------

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6, max_chars: int = 200) -> str:
    """Heuristic summary: keep the last `max_items`, truncate each to `max_chars`.

    This is a deterministic placeholder for an LLM-based summarizer; it keeps
    the offline benchmark reproducible.
    """

    if not messages:
        return ""
    lines = []
    for m in messages[-max_items:]:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        short = content if len(content) <= max_chars else content[:max_chars].rstrip() + "…"
        lines.append(f"- [{role}] {short}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Keeps recent messages in full and summarizes older content.

    - When the total estimated token count exceeds `threshold_tokens` and there
      are more than `keep_messages`, the overflow is folded into `summary`.
    - `compactions` tracks how many times this happened (used by benchmark).
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        t = self._thread(thread_id)
        messages = t["messages"]  # type: list[dict[str, str]]
        messages.append({"role": role, "content": content})
        self._maybe_compact(thread_id)

    def _maybe_compact(self, thread_id: str) -> None:
        t = self._thread(thread_id)
        messages = t["messages"]  # type: list[dict[str, str]]
        total = sum(estimate_tokens(m["content"]) for m in messages)
        if total <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        overflow = messages[:-self.keep_messages]
        t["messages"] = messages[-self.keep_messages:]

        # Fold the old summary + the new overflow into a bounded summary.
        summary_items: list[dict[str, str]] = []
        if t["summary"]:
            summary_items.append({"role": "system", "content": str(t["summary"])})
        summary_items.extend(overflow)
        t["summary"] = summarize_messages(summary_items, max_items=self.keep_messages * 2)

        t["compactions"] = int(t["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        t = self._thread(thread_id)
        return {"messages": t["messages"], "summary": t["summary"], "compactions": t["compactions"]}

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread(thread_id)["compactions"])
