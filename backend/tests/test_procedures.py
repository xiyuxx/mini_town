import asyncio
import json
import time
from unittest.mock import AsyncMock

import aiosqlite

from backend.town.engine import SimulationEngine
from backend.town.procedures import ProcedureStore


def experience(location, purpose="吃早餐", timestamp=0, resource="food_home_wang"):
    return {"purpose": purpose, "focus": purpose, "start_location": location,
            "completed_at": timestamp, "actions": [
                {"interaction_type": "inspect", "target_id": resource, "content": "查看食物"},
                {"interaction_type": "consume", "resource_id": resource, "content": "吃早餐",
                 "validated_effects": [{"type": "fake"}], "action_id": "old"},
            ]}


def test_thousand_skills_retrieve_old_relevant_summary_without_loading_bodies():
    async def scenario():
        engine = SimulationEngine()
        await engine.init()
        agent = engine.agents[0]
        store = engine.procedure_store
        old = await store.learn(agent.id, experience(agent.state.current_location))
        async with aiosqlite.connect(store.db_path) as db:
            for index in range(1000):
                item = experience(agent.state.current_location, purpose=f"维护设备{index}", timestamp=index + 1)
                item["actions"][1]["quantity"] = index + 1
                await store.learn(agent.id, item, db=db)
            await db.commit()
        engine.day = 3
        began = time.perf_counter()
        results = await store.discover(agent, engine, {"purpose": "吃早餐"})
        elapsed = time.perf_counter() - began
        assert results[0]["id"] == old
        assert len(results) <= 4
        assert all("actions" not in item and "embedding" not in item for item in results)
        assert len(json.dumps(results, ensure_ascii=False)) < 3000
        print(f"1001 procedures: discovery {elapsed:.3f}s, {len(results)} summaries")
        async with aiosqlite.connect(store.db_path) as db:
            count = await db.execute("SELECT count(*) FROM procedures")
            assert (await count.fetchone())[0] == 1001
        assert await store.load(engine.agents[1].id, old, engine.get_sim_timestamp()) is None
        assert await store.load(agent.id, old, -1) is None
        await engine._persist_cognition()
        async with aiosqlite.connect(store.db_path) as db:
            cursor = await db.execute("SELECT state_json FROM cognition_snapshots WHERE agent_id=?", (agent.id,))
            assert "experiences" not in json.loads((await cursor.fetchone())[0])
        await engine.shutdown()
    asyncio.run(scenario())


def test_structural_deduplication_keeps_distinct_methods_and_failure_feedback():
    async def scenario():
        engine = SimulationEngine()
        await engine.init()
        agent = engine.agents[0]
        store = engine.procedure_store
        original = experience(agent.state.current_location)
        key = await store.learn(agent.id, original)
        changed = experience(agent.state.current_location, "补充早餐", timestamp=10)
        changed["actions"][1].update(content="食用早餐", duration_minutes=20)
        assert await store.learn(agent.id, changed) == key
        other = experience(agent.state.current_location)
        other["actions"][0]["interaction_type"] = "request_service"
        assert await store.learn(agent.id, other) != key
        loaded = await store.load(agent.id, key, 10)
        assert "action_id" not in loaded["actions"][1]
        assert "validated_effects" not in loaded["actions"][1]
        await store.fail(agent.id, key, "库存不足", 10)
        assert await store.load(agent.id, key, 69) is None
        assert await store.load(agent.id, key, 70) is not None
        # Temporary shortages do not count toward structural archival.
        for now in (70, 130, 190):
            await store.fail(agent.id, key, "库存不足", now)
        assert await store.load(agent.id, key, 250) is not None
        for now in (250, 430, 610):
            await store.fail(agent.id, key, "设备不存在", now)
        assert await store.load(agent.id, key, 1000) is None
        assert await store.learn(agent.id, original) == key
        assert await store.load(agent.id, key, 1000) is not None
        await engine.shutdown()
    asyncio.run(scenario())


def test_legacy_snapshot_migration_is_atomic_and_idempotent(tmp_path):
    async def scenario():
        path = str(tmp_path / "legacy.db")
        item = experience("home_wang")
        item.update(id="legacy_skill", successes=7)
        async with aiosqlite.connect(path) as db:
            await db.execute("CREATE TABLE cognition_snapshots(agent_id TEXT PRIMARY KEY, state_json TEXT)")
            await db.execute("INSERT INTO cognition_snapshots VALUES (?, ?)",
                             ("wang", json.dumps({"goals": [], "experiences": [item]})))
            await db.commit()
        store = ProcedureStore(path)
        await store.init_db()
        await store.init_db()
        assert (await store.load("wang", "legacy_skill", 0))["actions"]
        async with aiosqlite.connect(path) as db:
            cursor = await db.execute("SELECT successes FROM procedures")
            assert (await cursor.fetchone())[0] == 7
            cursor = await db.execute("SELECT state_json FROM cognition_snapshots")
            assert json.loads((await cursor.fetchone())[0]) == {"goals": []}
    asyncio.run(scenario())


def test_semantic_retrieval_and_provider_failure_fallback():
    async def scenario():
        engine = SimulationEngine()
        await engine.init()
        agent = engine.agents[0]
        store = engine.procedure_store
        key = await store.learn(agent.id, experience(agent.state.current_location, "吃早餐"))
        provider = engine.embedding_provider
        provider.fallback = False
        provider.embed_one = AsyncMock(return_value=[1.0, 0.0])
        await store.index(key)
        results = await store.discover(agent, engine, {"purpose": "解决饥饿"})
        assert results[0]["id"] == key
        provider.embed_one = AsyncMock(side_effect=RuntimeError("offline"))
        results = await store.discover(agent, engine, {"purpose": "吃早餐"})
        assert results[0]["id"] == key
        await engine.shutdown()
    asyncio.run(scenario())


def test_reset_drains_pending_index_and_clears_procedural_memory():
    async def scenario():
        engine = SimulationEngine()
        await engine.init()
        agent = engine.agents[0]
        key = await engine.procedure_store.learn(agent.id, experience(agent.state.current_location))
        started = asyncio.Event()
        released = asyncio.Event()
        async def delayed_embedding(text):
            started.set()
            await released.wait()
            return [1.0]
        engine.embedding_provider.fallback = False
        engine.embedding_provider.embed_one = delayed_embedding
        engine._run_in_background(lambda: engine.procedure_store.index(key), "index", engine.get_sim_time_str())
        await started.wait()
        await engine.reset()
        assert not engine._background_tasks
        assert await engine.procedure_store.load(agent.id, key, engine.get_sim_timestamp()) is None
        await engine.shutdown()
    asyncio.run(scenario())
