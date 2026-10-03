from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config whose state lives under tmp_path.

    The compact threshold is lowered so compaction triggers quickly in tests.
    """
    root = Path(tmp_path)
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)

    model = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=root / "state",
        compact_threshold_tokens=60,
        compact_keep_messages=2,
        model=model,
        judge_model=model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(Path(tmp_path) / "profiles")

    # Default profile is returned before the file exists.
    default = store.read_text("dungct")
    assert default and "User Profile" in default

    # Write a custom profile and read it back.
    path = store.write_text("dungct", "# Hello\n")
    assert path.exists()
    assert store.read_text("dungct") == "# Hello\n"

    # edit_text replaces one occurrence and reports success.
    assert store.edit_text("dungct", "Hello", "Hi")
    assert "Hi" in store.read_text("dungct")

    # file_size reflects what is on disk.
    assert store.file_size("dungct") > 0

    # Structured facts are persisted inside the same User.md.
    store.upsert_fact("dungct", "name", "DũngCT")
    assert store.facts("dungct")["name"] == "DũngCT"


def test_compact_trigger(tmp_path: Path) -> None:
    cm = CompactMemoryManager(threshold_tokens=60, keep_messages=2)
    thread = "t1"
    long_msg = "lorem ipsum dolor sit amet consectetur adipiscing elit " * 4

    for _ in range(12):
        cm.append(thread, "user", long_msg)

    assert cm.compaction_count(thread) > 0
    ctx = cm.context(thread)
    assert len(ctx["messages"]) <= 2
    assert ctx["summary"]


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    # Teach both agents in one thread.
    baseline.reply("alice", "thread-a", "Mình tên là Lan, thích cà phê.")
    advanced.reply("alice", "thread-a", "Mình tên là Lan, thích cà phê.")

    # Ask in a *new* thread.
    advanced_ans = advanced.reply("alice", "thread-b", "Mình tên là gì?")["reply"]
    baseline_ans = baseline.reply("alice", "thread-b", "Mình tên là gì?")["reply"]

    assert "Lan" in advanced_ans
    assert "Lan" not in baseline_ans


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    long_msg = "Đây là một câu rất dài để test compact memory " * 10

    for _ in range(10):
        baseline.reply("bob", "t1", long_msg)
        advanced.reply("bob", "t1", long_msg)

    assert advanced.compaction_count("t1") > 0
    assert advanced.prompt_token_usage("t1") < baseline.prompt_token_usage("t1")
