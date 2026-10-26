# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vikingbot.agent.memory import MemoryStore
from vikingbot.openviking_mount.ov_server import VikingClient


class _FakeVikingClient:
    actor_peer_id = None

    def __init__(self):
        self.targets: list[str] = []

    def build_current_memory_target_uris(self, **_kwargs):
        return ["viking://user/default/peers/conv-26/memories/"]

    async def find(self, **kwargs):
        target = kwargs["target_uri"]
        self.targets.append(target)
        if target.endswith("/midterm_qa/"):
            return [
                {
                    "uri": f"{target}pair_hash.md",
                    "score": 0.91,
                    "abstract": "Caroline attended on 7 May 2023.",
                }
            ]
        return []

    async def read_content(self, uri, level="read"):
        del uri, level
        return "# Dialogue Pair\n\nCaroline: I attended on 7 May 2023."

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_midterm_vector_recall_has_independent_quota_and_group(tmp_path, monkeypatch):
    ov_server = SimpleNamespace(
        admin_user_id="default",
        memory_recall_events_limit=1,
        memory_recall_entities_limit=1,
        memory_recall_preferences_limit=1,
        memory_recall_max_chars=1000,
        memory_recall_midterm_enabled=True,
        memory_recall_midterm_limit=7,
        memory_recall_midterm_max_chars=1200,
        memory_recall_midterm_min_score=0.25,
    )
    store = MemoryStore(tmp_path, config=SimpleNamespace(ov_server=ov_server))
    client = _FakeVikingClient()
    monkeypatch.setattr(VikingClient, "create", AsyncMock(return_value=client))

    context = await store.get_viking_memory_context(
        current_message="When did Caroline attend?",
        workspace_id="agent",
        sender_id="conv-26",
    )

    assert any(target.endswith("/midterm_qa/") for target in client.targets)
    assert '<memory_group type="midterm_qa">' in context
    assert "7 May 2023" in context
