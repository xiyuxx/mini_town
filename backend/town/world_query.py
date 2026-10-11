"""Read-only, task-scoped queries over the current world state."""

from .world import LOCATION_MAP, bfs_path, can_enter


class WorldQuery:
    """Resolve an intention into a bounded view of the shared world."""

    def __init__(self, world_pack, resources):
        self.world_pack = world_pack
        self.resources = resources

    def locations_for_intention(self, agent, engine, intention: dict | list[dict] | None = None,
                                limit: int = 6) -> list[dict]:
        """Return accessible, reachable locations relevant to one or more intents."""
        intentions = intention if isinstance(intention, list) else [intention or {}]
        types = {str(item.get("interaction_type", "")) for item in intentions}
        targets = list(dict.fromkeys(str(item.get("target_location", "")) for item in intentions))
        resource_kinds = {str(item.get("resource_kind", "")) for item in intentions} - {""}
        agent_ids = {item.get("target_agent_id") for item in intentions}
        agent_locations = {item.state.current_location for item in engine.agents if item.id in agent_ids}
        ranked: list[tuple[float, dict]] = []
        for location in LOCATION_MAP.values():
            if not can_enter(agent.id, location.id):
                continue
            is_current = location.id == agent.state.current_location
            path = [] if is_current else bfs_path(
                agent.state.x, agent.state.y, *location.center,
                agent_id=agent.id, target_location_id=location.id,
            )
            if not is_current and not path:
                continue

            distance = abs(agent.state.x - location.center[0]) + abs(
                agent.state.y - location.center[1]
            )
            score = 0.25 if is_current else 0.0
            reasons: list[str] = []
            if is_current:
                reasons.append("当前地点")
            if location.id in targets:
                score += 3.0
                reasons.append("意图指定地点")
            if location.id in agent_locations:
                score += 2.0
                reasons.append("目标角色在此处")
            if any(self.resources.at_location(location.id, kind=kind) for kind in resource_kinds):
                score += 1.2
                reasons.append("有相关可用资源")
            if types & {"operate", "produce", "use_resource"}:
                processes = self.resources.processes_at(location.id)
                if processes:
                    score += 0.8
                    reasons.append("有可用过程")
            if types & {"communicate", "request_service"}:
                nearby = sum(
                    1 for other in engine.agents
                    if other.id != agent.id and other.state.current_location == location.id
                )
                if nearby:
                    score += min(0.8, nearby * 0.2)
                    reasons.append(f"有{nearby}位其他角色")
            score -= min(0.8, distance / 80)
            ranked.append((score, {
                "id": location.id,
                "name": location.name,
                "type": location.type,
                "accessible": True,
                "reachable": True,
                "distance": distance,
                "estimated_travel_minutes": len(path) * 5,
                "relevance": round(max(0.0, min(1.0, score / 4)), 3),
                "reasons": reasons,
            }))

        ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        views = {item["id"]: item for _, item in ranked}
        required = list(dict.fromkeys([*targets, *sorted(agent_locations), agent.state.current_location]))
        selected = [views[key] for key in required if key in views][:max(1, limit)]
        selected.extend(item for _, item in ranked if item not in selected)
        return selected[:max(1, limit)]

    def visible_environment(self, agent, engine, target_id: str = "") -> dict:
        """Return the existing observation snapshot as a world interface."""
        return engine.interactions.observation_snapshot(agent, engine, target_id)

    def current_constraints(self, agent, engine) -> list[dict]:
        """Return code-owned constraints needed before choosing an action."""
        constraints = []
        commitment = agent.get_current_commitment(engine.day, engine.hour, engine.minute)
        if commitment and commitment.get("status") not in {"completed", "skipped"}:
            constraints.append({"type": "commitment", "value": commitment})
        if not can_enter(agent.id, agent.state.current_location):
            constraints.append({"type": "location_access", "location": agent.state.current_location})
        weather = getattr(engine, "weather", {})
        if isinstance(weather, dict) and float(weather.get("travel_cost", 0.0)) > 0:
            constraints.append({
                "type": "travel_cost",
                "value": float(weather.get("travel_cost", 0.0)),
                "condition": weather.get("condition", ""),
            })
        return constraints
