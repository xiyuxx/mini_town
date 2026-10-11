"""Task-scoped read models assembled from runtime and durable state."""

import json
from collections import deque

from .considerations import collect_active_considerations
from .world_query import WorldQuery


class ContextService:
    """Build bounded LLM views without changing the complete runtime state."""

    _FIELDS = {
        "action_decision": {
            "world", "sim_time", "character", "duties", "schedule", "location",
            "inventory", "nearby_agents", "known_locations", "weather", "travel_cost",
            "mobility", "needs", "mental_state", "daily_plan", "routine_blocks",
            "known_facts", "recent_observation", "recent_interactions",
            "relevant_memories", "open_loops", "principles", "active_considerations",
        },
        "daily_plan": {
            "world", "sim_time", "character", "duties", "schedule", "location",
            "nearby_agents", "known_locations", "weather", "travel_cost", "mobility",
            "needs", "mental_state", "routine_blocks", "known_facts", "relevant_memories",
            "open_loops", "principles", "active_considerations",
        },
        "life_plan": {
            "world", "sim_time", "character", "duties", "schedule", "location",
            "inventory", "nearby_agents", "known_locations", "weather", "travel_cost",
            "mobility", "needs", "mental_state", "daily_plan", "routine_blocks",
            "known_facts", "recent_observation", "recent_interactions",
            "relevant_memories", "open_loops", "principles", "active_considerations",
        },
        "intention_proposal": {
            "world", "sim_time", "character", "duties", "schedule", "location",
            "nearby_agents", "weather", "travel_cost", "mobility", "needs",
            "mental_state", "daily_plan", "routine_blocks", "known_facts",
            "relevant_memories", "open_loops", "principles", "active_considerations",
        },
        "routine_selection": {
            "world", "sim_time", "character", "needs", "mental_state", "daily_plan",
            "nearby_agents", "relevant_memories", "active_considerations",
        },
        "dialogue": {
            "sim_time", "character", "location", "nearby_agents", "mental_state",
            "known_facts", "relevant_memories", "recent_interactions",
            "active_considerations",
        },
        "reflection": {
            "sim_time", "character", "mental_state", "relevant_memories",
            "recent_interactions", "active_considerations",
        },
        "memory_formation": {
            "sim_time", "character", "mental_state", "relevant_memories",
            "active_considerations",
        },
    }

    def __init__(self, world_pack, resources):
        self.world_query = WorldQuery(world_pack, resources)
        self._metrics = deque(maxlen=200)

    def recent_metrics(self, limit: int = 50) -> list[dict]:
        return list(self._metrics)[-max(1, min(limit, 200)):]

    async def build(self, agent, task_type: str, engine,
                    intention: dict | None = None,
                    sim_time_str: str | None = None) -> dict:
        """Build a task-specific view from the existing planning data source."""
        task_type = str(task_type or "action_decision")
        fields = self._FIELDS.get(task_type, self._FIELDS["action_decision"])
        current_intention = intention or self._current_intention(agent, engine)
        context = await agent._build_planning_context(
            sim_time_str or engine.get_sim_time_str(), engine, task_type=task_type,
            fields=fields, query=str(current_intention.get("purpose", "")).strip(),
        )
        now = engine.get_sim_timestamp()
        context["request"] = {
            "agent_id": agent.id,
            "task_type": task_type,
            "sim_timestamp": now,
            "state_version": engine.state_version,
            "intention_id": str((current_intention or {}).get("id", "")),
            "plan_id": str((current_intention or {}).get("plan_id", "")),
            "plan_step_id": str((current_intention or {}).get("plan_step_id", "")),
        }
        context["active_considerations"] = [
            item.to_dict() for item in collect_active_considerations(agent, engine)
        ]
        if "known_locations" in fields:
            intentions = self._daily_intentions(agent, engine) if task_type == "daily_plan" else current_intention
            context["known_locations"] = self.world_query.locations_for_intention(
                agent, engine, intentions, limit=6,
            )
        context["known_facts"] = list(context.get("known_facts", []))[-12:]
        context["routine_blocks"] = self._routine_view(agent, engine, task_type)
        result = {
            "request": context["request"],
            **{key: value for key, value in context.items() if key in fields},
        }
        self._metrics.append({
            "agentId": agent.id,
            "taskType": task_type,
            "simTimestamp": now,
            "stateVersion": engine.state_version,
            "intention": dict(current_intention or {}),
            "locationIds": [item["id"] for item in result.get("known_locations", [])],
            "omittedSections": sorted(set(context) - set(result)),
            "inputChars": len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))),
        })
        return result

    @staticmethod
    def _daily_intentions(agent, engine) -> list[dict]:
        """Cover the whole day's commitments, needs and flexible activities."""
        intentions = [{"target_location": item.location} for item in agent.schedule
                      if not item.is_flexible_slot]
        intentions.extend(
            {"target_location": location}
            for block in agent.routine_blocks for location in block.candidate_locations
        )
        intentions.extend(
            {"interaction_type": item.interaction_type, "target_location": item.target_location}
            for item in agent.mental_state.open_intentions(engine.get_sim_timestamp())
        )
        intentions.extend([{"resource_kind": "food"}, {"interaction_type": "rest"}])
        return intentions

    @staticmethod
    def _current_intention(agent, engine) -> dict:
        commitment = agent.get_current_commitment(engine.day, engine.hour, engine.minute)
        if commitment and commitment.get("status") not in {"completed", "skipped"}:
            return {
                "interaction_type": commitment.get("kind", "wait"),
                "target_location": commitment.get("location", ""),
            }
        step = agent.mental_state.life_plan.current_step if (
            agent.mental_state.life_plan and agent.mental_state.life_plan.status == "active"
        ) else None
        if step:
            return dict(step.action)
        intention = next(iter(agent.mental_state.open_intentions(
            engine.get_sim_timestamp()
        )), None)
        return {
            "interaction_type": intention.interaction_type if intention else "",
            "target_location": intention.target_location if intention else "",
        }

    @staticmethod
    def _routine_view(agent, engine, task_type: str) -> list[dict]:
        if task_type in {"daily_plan", "life_plan"}:
            return [block.to_dict() for block in agent.routine_blocks]
        return [
            block.to_dict()
            for block in agent.get_routine_blocks_for_time(engine.hour, engine.minute)
        ]
