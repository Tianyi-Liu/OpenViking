# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.message import Message, TextPart
from openviking.session.extraction_batch import (
    ExtractionBatchLimits,
    plan_extraction_batches,
    resolve_extraction_batch_limits,
)
from openviking.session.session import Session
from openviking.session.topic_continuity import TopicContinuityOptions, turn_text
from openviking_cli.utils.config.memory_config import MemoryConfig, TopicContinuityConfig


def _message(message_id: str, role: str = "user", text: str = "content") -> Message:
    return Message(id=message_id, role=role, parts=[TextPart(text)])


def _conversation(*turns: tuple[str, str]) -> list[Message]:
    messages: list[Message] = []
    for index, (user_text, assistant_text) in enumerate(turns, start=1):
        messages.append(_message(f"u{index}", text=user_text))
        messages.append(_message(f"a{index}", role="assistant", text=assistant_text))
    return messages


def _topic_enabled(**overrides) -> TopicContinuityOptions:
    values = {
        "enabled": True,
        "max_turns": 10,
        "overlap_turns": 1,
        "min_turns_to_split": 2,
        "skip_max_user_chars": 0,
    }
    values.update(overrides)
    return TopicContinuityOptions(**values)


def test_auto_commit_policy_resolves_phase2_batch_limits():
    limits = resolve_extraction_batch_limits(
        {
            "pending_token_threshold": 8000,
            "message_count_threshold": 40,
        }
    )

    assert limits == ExtractionBatchLimits(max_message_tokens=8000, max_messages=40)
    assert resolve_extraction_batch_limits(None).enabled is False
    assert resolve_extraction_batch_limits({}).enabled is False


def test_phase2_batches_split_an_oversized_batch_by_message_count():
    messages = [_message(f"m{index}") for index in range(5)]

    batches = plan_extraction_batches(
        messages,
        ExtractionBatchLimits(max_messages=2),
    )

    assert [[message.id for message in batch.messages] for batch in batches] == [
        ["m0", "m1"],
        ["m2", "m3"],
        ["m4"],
    ]


def test_phase2_batches_split_an_oversized_batch_by_token_count():
    messages = [_message(f"m{index}", text="word " * 80) for index in range(3)]
    one_message_tokens = messages[0].estimated_tokens

    batches = plan_extraction_batches(
        messages,
        ExtractionBatchLimits(max_message_tokens=one_message_tokens),
    )

    assert [[message.id for message in batch.messages] for batch in batches] == [
        ["m0"],
        ["m1"],
        ["m2"],
    ]
    assert all(batch.estimated_tokens <= one_message_tokens for batch in batches)


def test_phase2_batches_keep_a_turn_together_when_it_fits():
    messages = [
        _message("u1", text="question"),
        _message("a1", role="assistant", text="answer"),
        _message("u2", text="next"),
        _message("a2", role="assistant", text="done"),
    ]

    batches = plan_extraction_batches(
        messages,
        ExtractionBatchLimits(max_messages=2),
    )

    assert [[message.id for message in batch.messages] for batch in batches] == [
        ["u1", "a1"],
        ["u2", "a2"],
    ]


@pytest.mark.asyncio
async def test_working_memory_batches_carry_the_previous_summary_forward():
    vlm = type("VLM", (), {"is_available": lambda self: True})()
    session = Session(
        viking_fs=None,
        vlm_resolver=type(
            "Resolver",
            (),
            {"get_vlm": AsyncMock(return_value=vlm)},
        )(),
    )
    messages = [_message("u1"), _message("u2"), _message("u3")]
    calls = []

    async def generate(batch, latest_archive_overview="", checkpoint_requests=None):
        calls.append(
            (
                [message.id for message in batch],
                latest_archive_overview,
                list(checkpoint_requests or []),
            )
        )
        return f"{latest_archive_overview}|{batch[0].id}"

    session._generate_archive_summary_async = AsyncMock(side_effect=generate)

    limits = ExtractionBatchLimits(max_messages=1)
    result = await session._generate_archive_summary_with_batching(
        plan_extraction_batches(messages, limits),
        latest_archive_overview="previous",
        limits=limits,
    )

    assert result == "previous|u1|u2|u3"
    assert [(ids, previous) for ids, previous, _ in calls] == [
        (["u1"], "previous"),
        (["u2"], "previous|u1"),
        (["u3"], "previous|u1|u2"),
    ]


@pytest.mark.asyncio
async def test_working_memory_no_vlm_fallback_uses_all_messages(monkeypatch):
    messages = [_message("u1"), _message("u2"), _message("u3")]
    config = type("Config", (), {"vlm": None})()
    monkeypatch.setattr("openviking.session.session.get_openviking_config", lambda: config)
    session = Session(
        viking_fs=None,
        vlm_resolver=type(
            "Resolver",
            (),
            {"get_vlm": AsyncMock(return_value=None)},
        )(),
    )

    limits = ExtractionBatchLimits(max_messages=1)
    result = await session._generate_archive_summary_with_batching(
        plan_extraction_batches(messages, limits),
        latest_archive_overview="",
        limits=limits,
    )

    assert result == "# Session Summary\n\n**Overview**: 3 turns, 3 messages"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["create", "update", "fallback"])
@pytest.mark.parametrize(
    "message",
    [
        "Error code: 400 - InvalidParameter: Total tokens of multi-modal content "
        "and text exceed max message tokens.",
        "503 Service Unavailable",
    ],
)
async def test_working_memory_propagates_model_errors(monkeypatch, path, message):
    error = RuntimeError(message)
    completion = AsyncMock(side_effect=["no tool call", error] if path == "fallback" else error)
    vlm = SimpleNamespace(is_available=lambda: True, get_completion_async=completion)
    monkeypatch.setattr(
        "openviking.session.session.get_openviking_config",
        lambda: SimpleNamespace(output_language_override="en"),
    )
    session = Session(
        viking_fs=None,
        vlm_resolver=SimpleNamespace(get_vlm=AsyncMock(return_value=vlm)),
    )

    async def summarize():
        return await session._generate_archive_summary_async(
            [_message("u1")],
            latest_archive_overview="" if path == "create" else "## Current State\nPrevious work",
        )

    with pytest.raises(RuntimeError) as raised:
        await summarize()

    assert raised.value is error
    assert completion.await_count == (2 if path == "fallback" else 1)


@pytest.mark.asyncio
async def test_long_term_memory_batches_follow_the_planned_limit():
    session = Session(viking_fs=None)
    messages = [_message("u1"), _message("u2"), _message("u3")]
    recorded_batches = []

    async def extract(batch):
        return [batch[0].id]

    async def record(operation_name, step, batch, fn):
        result = await fn()
        recorded_batches.append((operation_name, step, [message.id for message in batch]))
        return result

    result = await session._extract_long_term_memories_with_batching(
        messages=messages,
        limits=ExtractionBatchLimits(max_messages=1),
        archive_uri="viking://user/default/sessions/s1/history/archive_001",
        extract_batch=extract,
        record_batch=record,
    )

    assert result == ["u1", "u2", "u3"]
    assert [message_ids for _, _, message_ids in recorded_batches] == [
        ["u1"],
        ["u2"],
        ["u3"],
    ]
    assert all(step == "long_term" for _, step, _ in recorded_batches)


@pytest.mark.asyncio
async def test_long_term_batching_flag_off_is_one_extract_call():
    session = Session(viking_fs=None)
    messages = _conversation(("q1", "a1"), ("q2", "a2"))
    recorded_batches = []

    async def extract(batch):
        return [message.id for message in batch]

    async def record(operation_name, step, batch, fn):
        result = await fn()
        recorded_batches.append([message.id for message in batch])
        return result

    await session._extract_long_term_memories_with_batching(
        messages=messages,
        limits=ExtractionBatchLimits(),
        archive_uri="viking://user/default/sessions/s1/history/archive_001",
        extract_batch=extract,
        record_batch=record,
    )
    assert recorded_batches == [["u1", "a1", "u2", "a2"]]


@pytest.mark.asyncio
async def test_long_term_batching_flag_on_calls_once_per_topic_group():
    session = Session(viking_fs=None)
    messages = _conversation(("login", "ok"), ("dinner", "noodles"))
    recorded = []
    extract_ids = []

    async def extract_and_store(batch):
        extract_ids.append([message.id for message in batch])
        return [message.id for message in batch]

    async def record(operation_name, step, batch, fn):
        result = await fn()
        recorded.append(
            {
                "owned": [message.id for message in batch],
                "operation": operation_name,
                "result": result,
            }
        )
        return result

    await session._extract_long_term_memories_with_batching(
        messages=messages,
        limits=ExtractionBatchLimits(),
        archive_uri="viking://user/default/sessions/s1/history/archive_001",
        extract_batch=extract_and_store,
        record_batch=record,
        topic=_topic_enabled(overlap_turns=1),
        continuity_check=lambda _previous, current: "dinner" not in turn_text(current),
    )
    assert extract_ids == [["u1", "a1"], ["u2", "a2"]]
    assert [item["owned"] for item in recorded] == [["u1", "a1"], ["u2", "a2"]]
    assert all(
        item["operation"].startswith("long_term_memory_extraction_batch_") for item in recorded
    )


@pytest.mark.asyncio
async def test_long_term_batching_records_owned_ids_only_with_overlap():
    session = Session(viking_fs=None)
    messages = _conversation(("t1", "a1"), ("t2", "a2"), ("t3", "a3"))
    owned_ids = []
    extract_ids = []

    async def extract(batch):
        extract_ids.append([message.id for message in batch])
        return []

    async def record(operation_name, step, batch, fn):
        owned_ids.append([message.id for message in batch])
        return await fn()

    await session._extract_long_term_memories_with_batching(
        messages=messages,
        limits=ExtractionBatchLimits(),
        archive_uri="viking://user/default/sessions/s1/history/archive_001",
        extract_batch=extract,
        record_batch=record,
        topic=_topic_enabled(max_turns=2, overlap_turns=1),
        continuity_check=lambda _previous, _current: True,
    )
    assert extract_ids == [["u1", "a1", "u2", "a2"], ["u2", "a2", "u3", "a3"]]
    assert owned_ids == [["u1", "a1", "u2", "a2"], ["u3", "a3"]]


@pytest.mark.asyncio
async def test_long_term_batching_with_policy_limits_after_topic_groups():
    session = Session(viking_fs=None)
    messages = _conversation(("same", "one"), ("same", "two"))
    extract_ids = []

    async def extract(batch):
        extract_ids.append([message.id for message in batch])
        return []

    async def record(operation_name, step, batch, fn):
        return await fn()

    await session._extract_long_term_memories_with_batching(
        messages=messages,
        limits=ExtractionBatchLimits(max_messages=2),
        archive_uri="viking://user/default/sessions/s1/history/archive_001",
        extract_batch=extract,
        record_batch=record,
        topic=_topic_enabled(),
        continuity_check=lambda _previous, _current: True,
    )
    assert extract_ids == [["u1", "a1"], ["u2", "a2"]]


class _RecordingCompressor:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def extract_long_term_memories(self, *, messages, **_kwargs):
        self.calls.append([message.id for message in messages])
        return []


class _TaskTracker:
    async def start(self, *_args, **_kwargs) -> None:
        return None

    async def complete(self, *_args, **_kwargs) -> None:
        return None

    async def fail(self, *_args, **_kwargs) -> None:
        return None


async def _run_phase2(
    monkeypatch,
    messages: list[Message],
    *,
    topic: TopicContinuityConfig | None = None,
) -> tuple[_RecordingCompressor, list[dict]]:
    compressor = _RecordingCompressor()
    session = Session(viking_fs=None, session_compressor=compressor)
    meta_updates: list[dict] = []
    memory = MemoryConfig()
    if topic is not None:
        memory.topic_continuity = topic
    config = SimpleNamespace(memory=memory)
    monkeypatch.setattr(
        "openviking.session.session.get_openviking_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "openviking_cli.utils.config.open_viking_config.get_openviking_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "openviking.service.task_tracker.get_task_tracker",
        lambda: _TaskTracker(),
    )

    async def _capture_meta(_archive_uri, updates, lease_ref=None):
        meta_updates.append(dict(updates))

    session._merge_archive_meta = _capture_meta
    session._merge_and_save_commit_meta = AsyncMock()
    await session._run_memory_extraction(
        task_id="task-topic",
        archive_uri="viking://user/default/sessions/s1/history/archive_001",
        messages=messages,
        first_message_id=messages[0].id,
        last_message_id=messages[-1].id,
        memory_policy={"working_memory": {"enabled": False}},
        auto_commit_policy=None,
    )
    return compressor, meta_updates


def _completed_long_term_ids(meta_updates: list[dict]) -> list[set[str]]:
    completed = []
    for update in meta_updates:
        steps = update.get("completed_memory_steps") or {}
        if "long_term" in steps:
            completed.append(set(steps["long_term"]))
    return completed


@pytest.mark.asyncio
async def test_commit_path_flag_off_extracts_once_without_policy(monkeypatch):
    messages = _conversation(("q1", "a1"), ("q2", "a2"))
    compressor, meta_updates = await _run_phase2(monkeypatch, messages)
    assert compressor.calls == [["u1", "a1", "u2", "a2"]]
    assert _completed_long_term_ids(meta_updates) == [{"u1", "a1", "u2", "a2"}]


@pytest.mark.asyncio
async def test_commit_path_flag_on_stays_one_batch_when_continuous(monkeypatch):
    messages = _conversation(("login", "ok"), ("still login", "fixed"))
    compressor, _meta_updates = await _run_phase2(
        monkeypatch,
        messages,
        topic=TopicContinuityConfig(enabled=True),
    )
    assert compressor.calls == [["u1", "a1", "u2", "a2"]]


@pytest.mark.asyncio
async def test_commit_path_flag_on_splits_on_max_turns_and_records_owned_ids(monkeypatch):
    messages = _conversation(("t1", "a1"), ("t2", "a2"), ("t3", "a3"))
    compressor, meta_updates = await _run_phase2(
        monkeypatch,
        messages,
        topic=TopicContinuityConfig(enabled=True, max_turns=2, overlap_turns=1),
    )
    assert compressor.calls == [
        ["u1", "a1", "u2", "a2"],
        ["u2", "a2", "u3", "a3"],
    ]
    completed = _completed_long_term_ids(meta_updates)
    assert completed[0] == {"u1", "a1", "u2", "a2"}
    assert completed[1] - completed[0] == {"u3", "a3"}
