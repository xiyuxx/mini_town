"""Agent: persona, state machine, perception, tool-based decision-making, and action execution."""

from datetime import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config import config
from .world import (
    location_center, location_name, find_location, LOCATION_MAP, agents_at_location,
    location_affordances, travel_cost, can_enter, GRID, GRID_W, GRID_H,
)
from .llm import LLMProvider
from .memory import MemoryStore, memories_to_text, auto_importance
from .tools_full import make_registry_for
from .schedule import RoutineBlock, ScheduleItem, compile_routine_blocks, compile_schedule
from .cognition import MentalState
from .daily_plan import DailyPlan
from .emotion import EmotionProfile

if TYPE_CHECKING:
    from .engine import SimulationEngine


@dataclass
class AgentState:
    id: str
    name: str
    age: int
    occupation: str
    personality: str
    background: str
    x: int
    y: int
    current_location: str
    current_action: str = "待机中"
    mood: str = "平静"
    emoji: str = "😐"
    status: str = "IDLE"  # IDLE | MOVING | ACTING | SPEAKING | LISTENING
    needs: dict[str, int] = field(default_factory=lambda: {
        "energy": 75,
        "hunger": 25,
        "social": 55,
    })

    def _action_progress(self) -> float | None:
        ticks = getattr(self, "_activity_ticks", 0)
        if self.status != "ACTING" or ticks <= 0:
            return None
        total = max(1, getattr(self, "_activity_total_ticks", ticks))
        return round(max(0.0, min(1.0, 1 - ticks / total)), 3)

    def _action_expected_duration(self) -> int | None:
        if self.status != "ACTING":
            return None
        return getattr(self, "_activity_total_ticks", 0) * config.TICK_INTERVAL_MINUTES

    def _emoji(self) -> str:
        if self.status == "MOVING":
            return "🚶"
        if self.status in ("SPEAKING", "LISTENING"):
            return "💬"
        return self.emoji or "😐"

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "age": self.age,
            "occupation": self.occupation, "personality": self.personality,
            "background": self.background, "x": self.x, "y": self.y,
            "currentLocation": self.current_location,
            "currentAction": self.current_action,
            "actionPhase": self.status,
            "actionTarget": getattr(self, "_movement_reason", "") if self.status == "MOVING" else "",
            "actionReason": getattr(self, "_action_reason", ""),
            "actionProgress": self._action_progress(),
            "actionExpectedDuration": self._action_expected_duration(),
            "mood": self.mood, "emoji": self._emoji(),
            "status": self.status,
            "needs": self.needs.copy(),
        }


# ── Agent class ──────────────────────────────────────────────

class Agent:
    """Town agent with state machine, perception, and LLM‑backed decisions."""

    # Valid state transitions: from → {to}
    STATE_TRANSITIONS: dict[str, set[str]] = {
        "IDLE":      {"MOVING", "ACTING", "SPEAKING", "LISTENING"},
        "MOVING":    {"IDLE"},
        "ACTING":    {"IDLE"},
        "SPEAKING":  {"IDLE"},
        "LISTENING": {"SPEAKING", "IDLE"},
    }

    def __init__(self, state: AgentState, schedule: list[tuple[int, int, str, str]],
                 llm: LLMProvider, memory: MemoryStore,
                 routine_blocks: list[dict] | None = None,
                 emotion_profile: EmotionProfile | None = None):
        self.state = state
        self.schedule: list[ScheduleItem] = compile_schedule(schedule)
        self.routine_blocks: list[RoutineBlock] = compile_routine_blocks(routine_blocks)
        self.llm = llm
        self.memory = memory
        self._movement_path: list[tuple[int, int]] | None = None
        self._movement_reason: str = ""
        self._movement_source: str = ""
        self._activity_ticks: int = 0
        self._activity_desc: str = ""
        self._activity_source: str = ""
        self._activity_started_at: int | None = None
        self._activity_action: dict | None = None
        self._activity_effects: list[dict] = []
        self._movement_action: dict | None = None
        self._pending_action: dict | None = None
        self._activity_commitment_id: str | None = None
        self._activity_total_ticks: int = 0
        self._action_reason: str = ""
        self._last_observation: dict | None = None
        self._commitment_status: dict[str, str] = {}
        self._interrupted_activity: dict | None = None
        # Scenario data assigns duties; the simulation never branches on an
        # occupation or building name.
        self.duties: list[dict] = []
        self.mental_state = MentalState()
        self.daily_plan: DailyPlan | None = None
        self.emotion_profile = emotion_profile or EmotionProfile()

    # ── Convenience properties ────────────────────────────────

    @property
    def id(self) -> str: return self.state.id

    @property
    def name(self) -> str: return self.state.name

    # ── State machine helpers ─────────────────────────────────

    def _transition(self, new_status: str):
        """Transition to new_status if valid; log and stay otherwise."""
        if new_status not in self.STATE_TRANSITIONS.get(self.state.status, set()):
            # Invalid transition — stay in current state
            return
        self.state.status = new_status

    async def start_activity(self, description: str, duration_minutes: int,
                             engine_ref: "SimulationEngine",
                             emoji: str | None = None,
                             commitment_id: str | None = None) -> int:
        """Start a validated, time-consuming activity."""
        description = (description or "").strip()
        if not description:
            raise ValueError("activity description cannot be empty")
        duration = max(config.TICK_INTERVAL_MINUTES, duration_minutes)
        ticks = max(1, (duration + config.TICK_INTERVAL_MINUTES - 1) // config.TICK_INTERVAL_MINUTES)
        self._activity_ticks = ticks
        self._activity_total_ticks = ticks
        self._activity_desc = description
        self._activity_started_at = engine_ref.get_sim_timestamp()
        self._activity_commitment_id = commitment_id
        if commitment_id:
            self._commitment_status[commitment_id] = "active"
        self._transition("ACTING")
        self.state.current_action = description
        if emoji:
            self.state.emoji = emoji
        self.state.mood = self.mental_state.emotion.label
        return ticks

    # ── Persona ───────────────────────────────────────────────

    def persona_text(self) -> str:
        return (
            f"姓名: {self.state.name}\n"
            f"年龄: {self.state.age}岁\n"
            f"职业: {self.state.occupation}\n"
            f"性格: {self.state.personality}\n"
            f"背景: {self.state.background}\n"
            f"当前位置: {location_name(self.state.current_location)}\n"
            f"当前心情: {self.state.mood}"
        )

    # ── Schedule ──────────────────────────────────────────────

    def get_routine_blocks_for_time(self, hour: int, minute: int) -> list[RoutineBlock]:
        return [block for block in self.routine_blocks if block.contains(hour, minute)]

    def get_schedule_for_time(self, hour: int, minute: int) -> str:
        current_minutes = hour * 60 + minute
        current_item = None
        next_item = None
        for item in self.schedule:
            if item.start_minutes <= current_minutes:
                current_item = item
            else:
                next_item = item
                break
        lines = []
        if current_item:
            affected = "、".join(current_item.affected_people) or "仅自己"
            lines.append(
                f"当前计划: {current_item.label}（在{location_name(current_item.location)}，"
                f"{current_item.hour:02d}:{current_item.minute:02d}开始；"
                f"柔性{current_item.flexibility:.2f}，责任{current_item.responsibility:.2f}，"
                f"影响对象：{affected}；延后后果：{current_item.consequence_of_delay}）"
            )
        if next_item:
            lines.append(
                f"接下来: {next_item.label}（在{location_name(next_item.location)}，"
                f"{next_item.hour:02d}:{next_item.minute:02d}）"
            )
        return "\n".join(lines) if lines else "没有特定安排"

    def _schedule_reason(self, hour: int = 6, minute: int = 0) -> str:
        item = self._get_schedule_item(hour, minute)
        return item.label if item else "出门"

    def _get_schedule_item(self, hour: int, minute: int) -> ScheduleItem | None:
        current_minutes = hour * 60 + minute
        current_item = None
        for item in self.schedule:
            if item.start_minutes <= current_minutes:
                current_item = item
            else:
                break
        return current_item

    def _get_schedule_location(self, hour: int, minute: int) -> str | None:
        item = self._get_schedule_item(hour, minute)
        return item.location if item else None

    def minutes_until_next_schedule(self, hour: int, minute: int) -> int | None:
        current = hour * 60 + minute
        for item in self.schedule:
            if item.start_minutes > current:
                return item.start_minutes - current
        return None

    def sync_schedule_goals(self, day: int, hour: int, minute: int) -> None:
        """Expose the current schedule item as a motive without forcing it."""
        current_minutes = hour * 60 + minute
        current_item = self._get_schedule_item(hour, minute)
        current_source = (
            f"schedule:{current_item.hour:02d}{current_item.minute:02d}"
            if current_item else ""
        )
        day_offset = (day - 1) * 1440
        for goal in self.mental_state.goals:
            if not goal.source.startswith("schedule:") or goal.status in {
                "completed", "failed", "abandoned"
            }:
                continue
            try:
                schedule_minutes = int(goal.source.split(":", 1)[1][:2]) * 60 + int(
                    goal.source.split(":", 1)[1][2:4]
                )
            except (ValueError, IndexError):
                continue
            goal.deadline = day_offset + schedule_minutes
            distance = schedule_minutes - current_minutes
            if goal.source == current_source:
                goal.status = "active"
                time_pressure = max(0.0, min(1.0, (30 - distance) / 30))
                goal.urgency = round(
                    max(0.1, time_pressure * goal.responsibility * (1.0 - goal.flexibility)), 2
                )
            elif schedule_minutes > current_minutes:
                goal.status = "proposed"
                goal.urgency = 0.1
            else:
                goal.status = "suspended"

    def get_current_commitment(self, day: int, hour: int, minute: int) -> dict | None:
        current_minutes = hour * 60 + minute
        item = self._get_schedule_item(hour, minute)
        if not item:
            return None
        commitment_id = f"{day}:{item.hour:02d}{item.minute:02d}"
        next_start = next(
            (scheduled.start_minutes for scheduled in self.schedule if scheduled.start_minutes > item.start_minutes),
            None,
        )
        return {
            "id": commitment_id,
            "label": item.label,
            "activity": item.activity,
            "kind": item.kind,
            "location": item.location,
            "start_minutes": item.start_minutes,
            "expected_duration": item.expected_duration,
            "deadline_minutes": next_start,
            "status": self._commitment_status.get(commitment_id, "pending"),
        }

    def complete_travel_commitment(self, commitment_id: str | None) -> None:
        if commitment_id:
            self._commitment_status[commitment_id] = "completed"

    def complete_goal_commitment(self, goal_id: str, day: int) -> str | None:
        """Keep schedule commitment state aligned with a completed schedule goal."""
        goal = next((item for item in self.mental_state.goals if item.id == goal_id), None)
        if not goal or goal.status != "completed" or not goal.source.startswith("schedule:"):
            return None
        slot = goal.source.split(":", 1)[1][:4]
        if len(slot) != 4 or not slot.isdigit():
            return None
        commitment_id = f"{day}:{slot}"
        self._commitment_status[commitment_id] = "completed"
        return commitment_id

    def complete_activity_commitment(self) -> str | None:
        commitment_id = self._activity_commitment_id
        self._activity_commitment_id = None
        if commitment_id:
            self._commitment_status[commitment_id] = "completed"
        return commitment_id

    def interrupt_activity(self) -> str:
        """Pause an activity so it can be resumed after a conversation."""
        description = self._activity_desc.strip()
        if description:
            self._interrupted_activity = {
                "description": description,
                "remaining_ticks": max(1, self._activity_ticks),
                "commitment_id": self._activity_commitment_id,
            }
        if self._activity_commitment_id:
            self._commitment_status[self._activity_commitment_id] = "interrupted"
        self._activity_ticks = 0
        self._activity_desc = ""
        self._activity_commitment_id = None
        self.state.status = "IDLE"
        return description

    def resume_interrupted_activity(self, current_commitment_id: str | None) -> bool:
        interrupted = self._interrupted_activity
        if not interrupted:
            return False
        commitment_id = interrupted.get("commitment_id")
        if commitment_id and commitment_id != current_commitment_id:
            self._commitment_status[commitment_id] = "skipped"
            self._interrupted_activity = None
            return False
        self._activity_desc = interrupted["description"]
        self._activity_ticks = interrupted["remaining_ticks"]
        self._activity_commitment_id = commitment_id
        if commitment_id:
            self._commitment_status[commitment_id] = "active"
        self.state.current_action = self._activity_desc
        self.state.status = "ACTING"
        self._interrupted_activity = None
        return True

    def reset_commitments_for_new_day(self):
        self._commitment_status.clear()

    # ── Perception ────────────────────────────────────────────

    def perceive(self, engine_ref: "SimulationEngine", as_stimulus: bool = False) -> dict:
        """Gather the world state visible to this agent.

        Called on arrival at a new location (MOVING → IDLE transition) and
        during movement steps (as_stimulus=True).

        When as_stimulus=True, returns a Stimulus dict:
          {type, location, description, emotional_valence, cell_type, weather, nearby_agents}
        """
        # ── Stimulus mode: per-step movement perception ──
        if as_stimulus:
            return self._perceive_stimulus(engine_ref)

        # ── Arrival mode: LLM context ──
        loc_id = self.state.current_location
        loc_name = location_name(loc_id)

        # Nearby agents (exclude self)
        nearby = agents_at_location(engine_ref.agents, loc_id)
        nearby_names = [a.name for a in nearby if a.id != self.id]

        # Weather
        weather = getattr(engine_ref, "weather", {})

        # Location info
        loc = LOCATION_MAP.get(loc_id)

        return {
            "location": loc_name,
            "location_id": loc_id,
            "location_type": loc.type if loc else "unknown",
            "nearby_agents": nearby_names,
            "weather": weather,
        }

    def _perceive_stimulus(self, engine_ref: "SimulationEngine") -> dict:
        """Per-step perception during movement: cell type, objects, nearby agents, weather.

        Returns a Stimulus dict with emotional valence for association scoring.
        """
        x, y = self.state.x, self.state.y

        # Cell type from grid
        cell_type = GRID[y][x] if 0 <= y < GRID_H and 0 <= x < GRID_W else "void"

        # Nearby agents within visual range (3 cells Manhattan)
        nearby_names: list[str] = []
        for a in engine_ref.agents:
            if a.id == self.id:
                continue
            if abs(a.state.x - x) + abs(a.state.y - y) <= 3:
                nearby_names.append(a.name)

        # Weather
        weather_raw = getattr(engine_ref, "weather", {})
        weather_text = (
            weather_raw.get("description", "") or weather_raw.get("condition", "")
            if isinstance(weather_raw, dict) else str(weather_raw)
        )

        # Location at current cell
        loc = find_location(x, y)
        loc_name = loc.name if loc else "路上"

        # Build natural language description
        parts = [f"在{loc_name}"]
        parts.append(loc.type if loc else cell_type)
        if nearby_names:
            parts.append(f"看到{'、'.join(nearby_names)}")
        parts.append(f"天气{weather_text}")
        description = "，".join(parts)

        return {
            "type": "movement_stimulus",
            "location": (x, y),
            "description": description,
            "emotional_valence": 0.0,
            "cell_type": cell_type,
            "weather": weather_text,
            "nearby_agents": nearby_names,
        }

    # ── Association (embedding-backed memory retrieval) ──────

    async def associate(self, stimulus: dict, embedding_provider, memory_store,
                        current_sim_timestamp: int | None = None) -> list[dict]:
        """Embed stimulus and retrieve semantically similar past memories.

        Returns list of {memory, similarity} dicts above similarity threshold (0.75),
        sorted by similarity descending.
        """
        description = stimulus.get("description", "")
        if not description:
            return []

        # Use semantic_search for embedding-backed retrieval
        try:
            memories = await memory_store.semantic_search(
                self.id, description, embedding_provider, limit=10,
                current_sim_timestamp=current_sim_timestamp,
            )
        except Exception:
            return []

        if not memories:
            return []

        # Compute stimulus embedding once for similarity scoring
        try:
            stim_emb = await embedding_provider.embed_one(description)
        except Exception:
            return []

        emotional_valence = stimulus.get("emotional_valence", 0.0)
        associations = []
        for mem in memories:
            if mem.embedding:
                sim = embedding_provider.cosine_similarity(stim_emb, mem.embedding)
            else:
                sim = 0.5
            if sim > 0.75:
                associations.append({
                    "memory": mem,
                    "similarity": sim,
                    "emotional_valence": emotional_valence,
                    "current_sim_timestamp": current_sim_timestamp,
                })
        associations.sort(key=lambda x: x["similarity"], reverse=True)
        return associations

    def should_interrupt(self, associations: list[dict], current_plan: str = "") -> bool:
        """Score associations to decide whether to interrupt the current plan.

        Weighted scoring per association:
            emotional_valence * 0.4 + similarity * 0.3 + age_days_factor * 0.2 + social_importance * 0.1
        Interrupt if max score > 0.7.

        Args:
            associations: list of {memory, similarity, emotional_valence} from associate()
            current_plan: current action description (unused in scoring, reserved)
        """
        if not associations:
            return False

        best_score = 0.0
        for assoc in associations:
            mem = assoc["memory"]
            similarity = assoc.get("similarity", 0.5)

            # Age factor uses simulation time, never wall-clock timestamps.
            current_timestamp = assoc.get("current_sim_timestamp")
            if current_timestamp is not None and mem.sim_timestamp is not None:
                age_days = max(0.0, (current_timestamp - mem.sim_timestamp) / 1440.0)
                age_factor = max(0.0, 1.0 - age_days / 30.0)
            else:
                age_factor = 0.5

            # Social importance: memory importance scaled to [0, 1]
            social_importance = mem.importance / 10.0

            # Emotional valence from the stimulus (included in association dict)
            valence = assoc.get("emotional_valence", 0.0)
            # Emotional valence: use absolute value — strong emotions (positive or
            # negative) drive interrupts more than neutral ones
            score = (
                abs(valence) * 0.4
                + similarity * 0.3
                + age_factor * 0.2
                + social_importance * 0.1
            )

            if score > best_score:
                best_score = score

        return best_score > 0.7

    # ── Decision ──────────────────────────────────────────────

    async def build_planning_context(self, sim_time_str: str,
                                     engine_ref: "SimulationEngine",
                                     task_type: str = "life_plan",
                                     intention: dict | None = None) -> dict:
        """Build a task-scoped view while retaining the complete mental state."""
        context_service = getattr(engine_ref, "context", None)
        if context_service is None:
            return await self._build_planning_context(sim_time_str, engine_ref)
        return await context_service.build(
            self, task_type, engine_ref, intention=intention,
            sim_time_str=sim_time_str,
        )

    async def decide_action(self, sim_time_str: str, engine_ref: "SimulationEngine") -> dict:
        """Compatibility entry point for conflict-only single-action decisions.

        Args:
            sim_time_str: "HH:MM" formatted simulation time.
            engine_ref: SimulationEngine for shared state access.

        Returns:
            Action dict: {"action": ..., "content": ..., "emoji": ..., "location"?: ...}
        """
        time_part = sim_time_str.split(" ")[-1]  # "第1天 06:00" → "06:00"
        hour, minute = map(int, time_part.split(":"))

        if self.llm.fallback:
            raise RuntimeError("decision model unavailable; no life action was invented")
        intention_context = await self.build_planning_context(
            sim_time_str, engine_ref, task_type="intention_proposal",
        )
        intention = await engine_ref.planner.propose_intention(
            self, engine_ref, intention_context,
        )
        planner_context = await self.build_planning_context(
            sim_time_str, engine_ref, task_type="action_decision",
            intention=intention,
        )
        return await engine_ref.planner.decide(self, engine_ref, planner_context)

    async def _build_planning_context(self, sim_time_str: str,
                                      engine_ref: "SimulationEngine",
                                      task_type: str = "life_plan",
                                      fields: set[str] | None = None,
                                      query: str = "") -> dict:
        time_part = sim_time_str.split(" ")[-1]
        hour, minute = map(int, time_part.split(":"))
        # Build perception context
        perception = self.perceive(engine_ref)
        schedule_text = self.get_schedule_for_time(hour, minute)
        if task_type == "daily_plan":
            schedule_text = "\n".join(
                f"{item.hour:02d}:{item.minute:02d} {item.label} · {item.location}"
                for item in self.schedule if not item.is_flexible_slot
            )
        current_commitment = self.get_current_commitment(engine_ref.day, hour, minute)
        if current_commitment and current_commitment.get("status") == "completed":
            schedule_text += "\n本时段的日程已经完成，可以做一项符合身份和地点的短暂自由活动，不要重复日程。"
        persona = self.persona_text()

        # Memories
        memories = await self.memory.retrieve(
            self.id, self.state.current_location,
            nearby_agents=perception["nearby_agents"],
            current_sim_timestamp=engine_ref.get_sim_timestamp(),
            query=query, limit=12,
        ) if fields is None or "relevant_memories" in fields else []
        mem_text = memories_to_text(memories)
        open_loops = await self.memory.get_open_loops(self.id) if (
            fields is None or "open_loops" in fields
        ) else []
        open_loop_text = "\n".join(
            f"- {memory.future_intention or memory.content}" for memory in open_loops
        ) or "（无）"

        now = engine_ref.get_sim_timestamp()
        mental_context = self.mental_state.summary_for(task_type, now)
        nearby_ids = {
            nearby.name: nearby.id for nearby in agents_at_location(
                engine_ref.agents, self.state.current_location
            ) if nearby.id != self.id
        }
        recent_interactions = {
            name: mental_context.get("recent_interactions", {}).get(agent_id)
            for name, agent_id in nearby_ids.items()
            if mental_context.get("recent_interactions", {}).get(agent_id)
        }
        weather_cost = travel_cost(perception["weather"])
        needs = self.state.needs
        need_context = {
            "energy": {"value": needs.get("energy", 0), "urgency": round(max(0, 25 - needs.get("energy", 0)) / 25, 2)},
            "hunger": {"value": needs.get("hunger", 0), "urgency": round(max(0, needs.get("hunger", 0) - 55) / 45, 2)},
            "social": {"value": needs.get("social", 0), "urgency": round(max(0, 35 - needs.get("social", 0)) / 35, 2)},
        }

        mobility_context = None
        if current_commitment:
            target_location = str(current_commitment.get("location", ""))
            target = LOCATION_MAP.get(target_location)
            distance = (
                abs(self.state.x - target.center[0]) + abs(self.state.y - target.center[1])
                if target else 0
            )
            travel_ticks = max(1, (distance + max(1, config.MOVE_SPEED) - 1) // max(1, config.MOVE_SPEED))
            estimated_travel_minutes = travel_ticks * config.TICK_INTERVAL_MINUTES
            start_minutes = int(current_commitment.get("start_minutes", hour * 60 + minute))
            latest_departure = start_minutes - estimated_travel_minutes
            current_minutes = hour * 60 + minute
            mobility_context = {
                "commitment_id": current_commitment.get("id"),
                "commitment_status": current_commitment.get("status"),
                "target_location_id": target_location,
                "target_location_name": location_name(target_location),
                "current_location_id": self.state.current_location,
                "travel_required": bool(target_location and target_location != self.state.current_location),
                "approx_distance": distance,
                "estimated_travel_minutes": estimated_travel_minutes,
                "scheduled_start_minutes": start_minutes,
                "latest_departure_minutes": latest_departure,
                "departure_delay_minutes": max(0, current_minutes - latest_departure),
            }

        known_facts = [
            {
                "id": fact.id, "time": fact.sim_time, "type": fact.type,
                "location": fact.location, "details": fact.details,
            }
            for fact in engine_ref.fact_ledger.known_for(self.id, limit=12)
            if fact.sim_timestamp <= now
        ] if fields is None or "known_facts" in fields else []
        planner_context = {
            "world": {
                "id": engine_ref.world_pack.id,
                "name": engine_ref.world_pack.name,
                "setting": engine_ref.world_pack.setting,
            },
            "sim_time": sim_time_str,
            "character": persona,
            "duties": list(self.duties),
            "schedule": schedule_text,
            "location": {
                "id": self.state.current_location,
                "description": perception["location"],
                "entities": engine_ref.resources.visible_at(
                    self.state.current_location, self.id
                ),
                "anchors": [anchor.to_dict() for anchor in engine_ref.resources.scene.anchors_at(
                    self.state.current_location
                )],
                "containers": [container.to_dict() for container in engine_ref.resources.scene.containers_at(
                    self.state.current_location
                )],
                "available_processes": engine_ref.resources.processes_at(
                    self.state.current_location
                ),
            } if fields is None or "location" in fields else {},
            "inventory": engine_ref.resources.inventory(self.id) if fields is None or "inventory" in fields else [],
            "nearby_agents": [
                {
                    "id": nearby.id, "name": nearby.name,
                    "status": nearby.state.status,
                    "current_action": nearby.state.current_action,
                }
                for nearby in agents_at_location(
                    engine_ref.agents, self.state.current_location
                ) if nearby.id != self.id
            ],
            # WorldQuery fills task-relevant locations after an intention is known.
            "known_locations": [],
            "weather": perception["weather"],
            "travel_cost": weather_cost,
            "mobility": mobility_context,
            "needs": need_context,
            "mental_state": mental_context,
            "daily_plan": self.daily_plan.to_dict() if self.daily_plan else None,
            "routine_blocks": [block.to_dict() for block in self.routine_blocks] if (
                fields is None or "routine_blocks" in fields
            ) else [],
            "known_facts": known_facts,
            "recent_observation": dict(self._last_observation) if self._last_observation else None,
            "recent_interactions": recent_interactions,
            "relevant_memories": mem_text,
            "open_loops": open_loop_text,
            "principles": [
                "日程是目标和约束，不是命令",
                "未知资源必须先检查，不能凭空使用",
                "recent_observation已经包含观察结果；同地点同目标没有变化时不要重复inspect",
                "改变公开意图时要引用新事实并显式更新状态",
                "每次只执行一个外显动作",
                "mobility.travel_required为true时，候选中必须包含move；若已晚于latest_departure，选择非移动行动必须有可引用的更高优先级事实或目标，不能用等待、观察、热身等填充动作逃避出发责任",
            ],
        }
        return {key: value for key, value in planner_context.items() if fields is None or key in fields}

    # ── Execution ─────────────────────────────────────────────

    async def execute_action(self, action: dict, engine_ref: "SimulationEngine") -> dict | None:
        """Execute the decided action, updating the state machine.

        Returns an event dict or None if nothing publishable happened.
        """
        action_type = action.get("action", "act")
        content = (action.get("content") or "").strip()
        loc_id = action.get("location")
        target_id = action.get("target")
        emoji = action.get("emoji", "😐")
        commitment = action.get("commitment") or {}
        commitment_id = commitment.get("id")

        self.state.emoji = emoji
        self._action_reason = str(action.get("selection_reason") or action.get("decision_reason") or "")[:240]

        # ── Move (BFS pathfinding, fallback path) ──
        if action_type == "move" and loc_id and loc_id in LOCATION_MAP:
            from .world import bfs_path
            cx, cy = location_center(loc_id)
            path = bfs_path(
                self.state.x, self.state.y, cx, cy,
                agent_id=self.id,
                target_location_id=loc_id,
            )
            if not path:
                self.state.current_action = "无法到达"
                return None
            self._movement_path = path
            self._movement_action = dict(action)
            self._movement_reason = content or self._schedule_reason(engine_ref.hour, engine_ref.minute)
            self._movement_source = action.get("source", "llm")
            self._transition("MOVING")
            self.state.current_action = f"前往{location_name(loc_id)}"
            if commitment_id and commitment.get("kind") == "mixed":
                pending_content = (commitment.get("activity") or "").strip()
                if pending_content:
                    self._pending_action = {
                        "action": "act",
                        "content": pending_content,
                        "duration_minutes": commitment.get("expected_duration"),
                        "commitment": commitment,
                        "source": action.get("source", "commitment"),
                    }
            elif commitment_id and commitment.get("kind") == "travel":
                self._pending_action = {"action": "complete_travel", "commitment": commitment}
            reason = content if content and content != "按日程前往" else self._schedule_reason()
            return {
                "type": "move_start", "agentIds": [self.id],
                "location": loc_id,
                "content": f"{self.name} · {location_name(loc_id)}",
                "cause": action.get("source", "llm"),
                "outcome": "已开始寻路",
            }

        # ── Act ──
        if action_type == "act":
            # Intent routing: act at a different location → move there first
            if loc_id and loc_id in LOCATION_MAP and loc_id != self.state.current_location:
                self._pending_action = {
                    **action,
                    "action": "act", "content": content, "emoji": emoji,
                    "duration_minutes": action.get("duration_minutes"),
                    "commitment": commitment,
                    "source": action.get("source", "llm"),
                }
                from .world import bfs_path
                cx, cy = location_center(loc_id)
                path = bfs_path(
                    self.state.x, self.state.y, cx, cy,
                    agent_id=self.id,
                    target_location_id=loc_id,
                )
                if path:
                    self._movement_path = path
                    self._movement_action = dict(action)
                    self._movement_reason = content
                    self._movement_source = action.get("source", "llm")
                    self._transition("MOVING")
                    self.state.current_action = f"前往{location_name(loc_id)}——{content}"
                    return {"type": "move_start", "agentIds": [self.id],
                            "location": loc_id,
                            "content": f"{self.name} · {location_name(loc_id)}",
                            "cause": action.get("source", "llm"),
                            "outcome": "已开始寻路"}
                return None  # no path

            if not content:
                return None
            if commitment_id:
                duration = max(
                    config.TICK_INTERVAL_MINUTES,
                    int(commitment.get("expected_duration") or action.get("duration_minutes") or config.TICK_INTERVAL_MINUTES),
                )
            else:
                duration = engine_ref.town_agent.estimate_activity_duration(
                    self, content, self.state.current_location,
                    engine_ref.hour, engine_ref.minute,
                    requested_minutes=action.get("duration_minutes"),
                )
            until_next = self.minutes_until_next_schedule(engine_ref.hour, engine_ref.minute)
            if until_next is not None and not commitment_id:
                duration = min(duration, max(config.TICK_INTERVAL_MINUTES, until_next))
            self._activity_source = action.get("source", "llm")
            self._activity_action = dict(action)
            self._activity_effects = list(action.get("validated_effects", []))
            await self.start_activity(content, duration, engine_ref, emoji, commitment_id)
            return {"type": "activity_started", "agentIds": [self.id],
                    "location": self.state.current_location,
                    "content": f"{self.name}开始{content}",
                    "cause": action.get("source", "llm"),
                    "outcome": f"预计持续{duration}分钟"}

        # ── Talk (trigger dialogue) ──
        if action_type == "talk" and target_id:
            target = next((a for a in engine_ref.agents if a.id == target_id), None)
            if not target:
                return None
            if target.state.current_location != self.state.current_location:
                self.state.current_action = f"想和{target.name}说话，但不在同一地点"
                return None
            self._transition("SPEAKING")
            self._transition("IDLE")
            return {
                "type": "dialogue_intent", "agentIds": [self.id, target_id],
                "location": self.state.current_location,
                "content": content,
                "initiator": self.id,
                "target": target_id,
            }

        # ── Reflect ──
        if action_type == "reflect":
            description = content or "静静思考了一会儿"
            duration = engine_ref.town_agent.estimate_activity_duration(
                self, description, self.state.current_location,
                engine_ref.hour, engine_ref.minute,
            )
            self._activity_action = dict(action)
            self._activity_effects = list(action.get("validated_effects", []))
            await self.start_activity(description, duration, engine_ref, emoji)
            return {"type": "activity_started", "agentIds": [self.id],
                    "location": self.state.current_location,
                    "content": f"{self.name}开始{description}"}

        # Unsupported actions are reported as blocked by the engine.
        return None
