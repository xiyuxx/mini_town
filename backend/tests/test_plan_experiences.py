import asyncio

import pytest

from backend.town.cognition import LifePlan, PlanStep, MentalState
from backend.town.engine import SimulationEngine


def test_successful_plan_is_learned_restored_and_revalidated(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.memory.db_path = str(tmp_path / "experience.db")
        await engine.init()
        agent = engine.agents[0]
        food = engine.resources.get(f"food_{agent.state.current_location}")
        assert food is not None
        agent.mental_state.life_plan = LifePlan(
            "吃早餐", "补充食物", engine.get_sim_timestamp(), engine.get_sim_timestamp() + 60,
            purpose="吃早餐", start_location=agent.state.current_location,
            steps=[
                PlanStep("查看食物", {"interaction_type": "inspect", "target_id": food.id,
                                    "content": "查看食物", "duration_minutes": 5}),
                PlanStep("吃早餐", {"interaction_type": "consume", "resource_id": food.id,
                                   "content": "吃早餐", "duration_minutes": 5}),
            ],
        )
        while agent.mental_state.life_plan.status == "active":
            action = engine.planner.next_plan_action(agent, engine)
            await engine._execute_decision(agent, action, engine.get_sim_time_str())
            while agent.state.status == "ACTING":
                engine.minute += 5
                await engine._process_acting(agent, engine.get_sim_time_str())
        learned_summary = (await engine.procedure_store.discover(
            agent, engine, {"purpose": "吃早餐"},
        ))[0]
        learned = await engine.procedure_store.load(agent.id, learned_summary["id"], engine.get_sim_timestamp())
        assert learned_summary["purpose"] == "吃早餐"
        assert "validated_effects" not in learned["actions"][1]
        assert "action_id" not in learned["actions"][1]
        assert learned["actions"][0]["target_id"] == food.id
        await engine.shutdown()

        restored = SimulationEngine()
        restored.memory.db_path = engine.memory.db_path
        await restored.init()
        agent = restored.agents[0]
        assert await restored.procedure_store.load(agent.id, learned["id"], engine.get_sim_timestamp()) == learned
        restored.hour, restored.minute = engine.hour, engine.minute
        restored.llm.fallback = False
        calls = []

        async def json_call(system, payload, mode="chat", task="json_call"):
            calls.append(task)
            if task == "intention_proposal":
                return {"purpose": "吃早餐", "interaction_type": "consume", "resource_kind": "food"}
            assert payload["experiences"][0]["id"] == learned["id"]
            assert "actions" not in payload["experiences"][0]
            return {"experience_id": learned["id"]}

        restored.planner._json_call = json_call
        context = await agent.build_planning_context(
            restored.get_sim_time_str(), restored, task_type="intention_proposal",
        )
        plan = await restored.planner.create_life_plan(agent, restored, context)
        assert calls == ["intention_proposal", "life_plan"]
        assert plan.experience_id == learned["id"]
        assert plan.steps[0].id != learned["id"]
        assert all(step.status == "pending" and not step.expected_outcome for step in plan.steps)
        agent.mental_state.life_plan = plan
        action = restored.planner.next_plan_action(agent, restored)
        await restored._execute_decision(agent, action, restored.get_sim_time_str())
        while agent.state.status == "ACTING":
            restored.minute += 5
            await restored._process_acting(agent, restored.get_sim_time_str())
        restored.resources.get(food.id).quantity = 0
        with pytest.raises(RuntimeError, match="blocked"):
            restored.planner.next_plan_action(agent, restored)
        await restored._block_plan(agent, plan, "", "资源数量不足")
        assert not await restored.procedure_store.discover(agent, restored, {"purpose": "吃早餐"})
        await restored.reset()
        assert not await restored.procedure_store.load(agent.id, learned["id"], restored.get_sim_timestamp())
        await restored.shutdown()

    asyncio.run(scenario())


def test_incomplete_and_trivial_plans_do_not_create_experiences():
    mental = MentalState()
    mental.life_plan = LifePlan("等待", "等待", 0, 60, steps=[
        PlanStep("等待", {"interaction_type": "wait"}),
        PlanStep("等待", {"interaction_type": "wait"}),
    ])
    mental.life_plan.start_current_step(0)
    assert mental.complete_plan_step(mental.life_plan.current_step.id, 5) is None
    mental.life_plan.start_current_step(5)
    assert mental.complete_plan_step(mental.life_plan.current_step.id, 10) is None
