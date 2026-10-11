import asyncio
from unittest.mock import AsyncMock

from backend.town.candidates import build_routine_candidates
from backend.town.engine import SimulationEngine


def test_context_queries_memory_once_and_skips_unused_world_sections():
    async def scenario():
        engine = SimulationEngine()
        await engine.init()
        agent = engine.agents[0]
        retrieve = AsyncMock(wraps=engine.memory.retrieve)
        engine.memory.retrieve = retrieve
        summary = agent.mental_state.summary_for
        summaries = []
        def tracked_summary(task, now):
            summaries.append(task)
            return summary(task, now)
        agent.mental_state.summary_for = tracked_summary
        context = await engine.context.build(agent, "life_plan", engine,
                                             intention={"purpose": "吃早餐"})
        assert "relevant_memories" in context
        assert retrieve.await_count == 1
        assert retrieve.call_args.kwargs["query"] == "吃早餐"
        assert summaries == ["life_plan"]
        def unused(*args, **kwargs):
            raise AssertionError("routine selection should not query location, inventory or open loops")
        engine.resources.visible_at = unused
        engine.resources.inventory = unused
        engine.memory.get_open_loops = unused
        engine.context.world_query.locations_for_intention = unused
        await engine.context.build(agent, "routine_selection", engine)
        await engine.shutdown()
    asyncio.run(scenario())


def test_daily_plan_covers_all_commitments_without_intention_call():
    async def scenario():
        engine = SimulationEngine()
        await engine.init()
        engine.llm.fallback = False
        agent = next(item for item in engine.agents if item.id == "mei")
        context = await engine.context.build(agent, "daily_plan", engine)
        locations = {item["id"] for item in context["known_locations"]}
        for item in agent.schedule:
            if not item.is_flexible_slot:
                assert item.location in locations
                assert f"{item.hour:02d}:{item.minute:02d}" in context["schedule"]
        calls = []
        async def json_call(system, payload, mode="chat", task="json_call"):
            calls.append(task)
            return {"focus": "安排一天", "blocks": [{"id": "morning", "window": [360, 720],
                                                    "intent": "完成上午安排", "candidates": []}]}
        engine.planner._json_call = json_call
        await engine.planner.create_daily_plan(agent, engine, context)
        assert calls == ["daily_plan"]
        await engine.shutdown()
    asyncio.run(scenario())


def test_routine_selection_uses_task_context_and_rejects_invented_candidate():
    async def scenario():
        engine = SimulationEngine()
        await engine.init()
        agent = next(item for item in engine.agents if item.id == "mei")
        engine.hour = 18
        candidates = build_routine_candidates(agent, engine)
        async def json_call(system, payload, mode="chat", task="json_call"):
            assert payload["request"]["task_type"] == "routine_selection"
            assert "active_considerations" in payload and "relevant_memories" in payload
            assert "inventory" not in payload and "known_locations" not in payload
            return {"candidate_id": candidates[0].id}
        engine.planner._json_call = json_call
        result = await engine.planner.select_routine_candidate(agent, engine, candidates)
        assert result["location"] == candidates[0].action["location"]
        await engine.shutdown()
    asyncio.run(scenario())
