from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    """Fraction of expected substrings that appear in the answer."""
    if not expected:
        return 1.0
    answer_lower = (answer or "").lower()
    hits = sum(1 for e in expected if e.lower() in answer_lower)
    return hits / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline mode.

    Combines fact coverage with a coarse "naturalness" signal so the baseline's
    "I don't remember" replies score clearly lower than a factual recall.
    """
    if not answer or not answer.strip():
        return 0.0

    coverage = recall_points(answer, expected)

    answer_lower = answer.lower()
    if any(x in answer_lower for x in ("chưa có thông tin", "không nhớ", "không có trí nhớ", "chưa lưu")):
        natural = 0.2
    elif len(answer) < 5:
        natural = 0.4
    elif len(answer) > 1000:
        natural = 0.7
    else:
        natural = 1.0

    return round(coverage * 0.7 + natural * 0.3, 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    memory_bytes = 0
    compactions = 0

    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = conv["id"]

        # 1. Feed all conversation turns (short-term memory / compaction live here).
        for turn in conv["turns"]:
            r = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += r.get("agent_tokens", 0)
            total_prompt_tokens += r.get("prompt_tokens", 0)

        # 2. Ask recall questions in a *fresh* thread to test cross-session memory.
        for i, q in enumerate(conv.get("recall_questions", [])):
            qid = f"{conv['id']}-recall-{i}"
            a = agent.reply(user_id, qid, q["question"])
            answer = a["reply"]
            total_agent_tokens += a.get("agent_tokens", 0)
            total_prompt_tokens += a.get("prompt_tokens", 0)
            recall_scores.append(recall_points(answer, q["expected_contains"]))
            quality_scores.append(heuristic_quality(answer, q["expected_contains"]))

        # 3. Record memory file growth (advanced only) and compaction count.
        if hasattr(agent, "memory_file_size"):
            memory_bytes = max(memory_bytes, agent.memory_file_size(user_id))
        if hasattr(agent, "compaction_count"):
            compactions += agent.compaction_count(thread_id)

    n = len(recall_scores)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(sum(recall_scores) / n, 3) if n else 0.0,
        response_quality=round(sum(quality_scores) / n, 3) if n else 0.0,
        memory_growth_bytes=memory_bytes,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    data = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(data, headers=headers, tablefmt="github")
    except Exception:
        lines = [" | ".join(headers), " | ".join("-" * len(h) for h in headers)]
        for row in data:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)


def _analysis() -> str:
    return "\n".join(
        [
            "## Phân tích kết quả",
            "- Advanced có recall tốt hơn Baseline vì nó ghi facts ổn định vào `User.md` (persistent memory), nên sang thread mới vẫn trả lời được; Baseline chỉ có short-term memory nên quên ngay khi sang thread mới.",
            "- Ở hội thoại ngắn Advanced có thể tốn token hơn Baseline vì mỗi lượt nó vẫn phải kéo `User.md` vào prompt (chi phí cố định của persistent memory), trong khi Baseline không có lớp này.",
            "- Ở hội thoại rất dài, compact memory giúp Advanced chỉ kéo summary + vài message gần nhất thay vì toàn bộ lịch sử, nên `Prompt tokens processed` thấp hơn hẳn Baseline (Baseline kéo nguyên vẹn mọi message cũ).",
            "- File `User.md` tăng trưởng theo thời gian; rủi ro là nó phình to (kéo prompt cost) và có thể lưu nhầm fact nếu extraction sai hoặc không xử lý correction nhiễu.",
        ]
    )


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")

    print("# Standard Benchmark")
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)
    print(
        format_rows(
            [
                run_agent_benchmark("Baseline", baseline, standard, config),
                run_agent_benchmark("Advanced", advanced, standard, config),
            ]
        )
    )

    print("\n# Long-Context Stress Benchmark")
    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)
    print(
        format_rows(
            [
                run_agent_benchmark("Baseline", baseline_stress, stress, config),
                run_agent_benchmark("Advanced", advanced_stress, stress, config),
            ]
        )
    )

    print("\n" + _analysis())


if __name__ == "__main__":
    main()
