# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from dataclasses import replace

from openviking.message import Message, TextPart
from openviking.session.midterm_memory import (
    build_midterm_records,
    deserialize_midterm_manifest,
    records_to_extraction_messages,
    render_midterm_record,
    render_midterm_recall_record,
    serialize_midterm_manifest,
)


def _message(
    index: int,
    text: str,
    *,
    speaker: str | None = None,
    role: str = "user",
    peer_id: str = "conv-26",
) -> Message:
    return Message(
        id=f"msg-{index}",
        role=role,
        parts=[TextPart(text)],
        peer_id=peer_id,
        speaker=speaker,
        created_at=f"2023-04-02T09:36:{index:02d}+00:00",
    )


def test_builds_pairs_from_consecutive_structured_speaker_blocks():
    records = build_midterm_records(
        [
            _message(1, "I am going to Paris.", speaker="Alice"),
            _message(2, "The ticket is booked.", speaker="Alice"),
            _message(3, "Is it for the exhibition?", speaker="Bob"),
            _message(4, "Do you need a hotel?", speaker="Bob"),
            _message(5, "Yes, please.", speaker="Alice"),
        ],
        session_id="session-1",
        archive_uri="viking://session/archive_001",
    )

    assert [record.kind for record in records] == ["dialogue_pair", "single_turn"]
    assert records[0].blocks[0].speaker == "Alice"
    assert records[0].blocks[0].texts == (
        "I am going to Paris.",
        "The ticket is booked.",
    )
    assert records[0].blocks[1].speaker == "Bob"
    assert records[1].source_message_ids == ("msg-5",)
    assert records[0].filename == "pair_0001.md"
    assert records[1].filename == "single_turn_0002.md"


def test_legacy_locomo_prefix_is_only_used_when_multiple_speakers_are_present():
    records = build_midterm_records(
        [
            _message(1, "Caroline: I joined the support group."),
            _message(2, "Melanie: I am proud of you."),
        ],
        session_id="session-1",
        archive_uri="viking://session/archive_001",
    )

    assert [block.speaker for block in records[0].blocks] == ["Caroline", "Melanie"]
    assert records[0].blocks[0].texts == ("I joined the support group.",)


def test_manifest_round_trip_drives_extraction_messages():
    records = build_midterm_records(
        [
            _message(1, "Question", speaker="Alice"),
            _message(2, "Answer", speaker="Bob"),
        ],
        session_id="session-1",
        archive_uri="viking://session/archive_001",
    )

    persisted = deserialize_midterm_manifest(serialize_midterm_manifest(records))
    extraction_messages = records_to_extraction_messages(persisted)

    assert [message.speaker for message in extraction_messages] == ["Alice", "Bob"]
    assert [message.content for message in extraction_messages] == ["Question", "Answer"]
    assert extraction_messages[0].source_message_ids == ["msg-1"]
    assert persisted[0].converted_to_long_term is False
    assert persisted[0].converted_at is None
    rendered = render_midterm_record(persisted[0])
    assert "Content-Hash:" in rendered
    assert rendered.count("Timestamp:") == 1
    assert "Timestamp: 2023-04-02T09:36:01+00:00" in rendered
    assert "Converted-To-Long-Term: false" in rendered


def test_content_hash_and_filename_are_deterministic():
    messages = [
        _message(1, "Question", speaker="Alice"),
        _message(2, "Answer", speaker="Bob"),
    ]
    first = build_midterm_records(
        messages,
        session_id="session-1",
        archive_uri="viking://session/archive_001",
    )[0]
    second = build_midterm_records(
        [replace(message, id=f"rerun-{message.id}") for message in messages],
        session_id="session-1",
        archive_uri="viking://session/archive_001",
    )[0]

    assert first.content_hash == second.content_hash
    assert first.filename == second.filename
    assert first.recall_filename == second.recall_filename
    assert first.peer_ids == ("conv-26",)
    first_recall = render_midterm_recall_record(first)
    second_recall = render_midterm_recall_record(second)
    assert first_recall == second_recall
    assert "session-1" not in first_recall
    assert "archive_001" not in first_recall
    assert first_recall.count("Timestamp:") == 1
    assert "Timestamp: 2023-04-02T09:36:01+00:00" in first_recall
