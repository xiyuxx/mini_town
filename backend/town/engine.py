"""Simulation engine: time loop, state machines, group dialogue, relationships, trace.

Phase 1 rewrite: integrates TownAgent, state-machine agents, group-chat dialogue,
relationship tracking on every interaction, and structured trace logging.
"""

import asyncio
import re
import uuid
import random
from datetime import datetime
from ..config import config
from .world import LOCATIONS, location_name, agents_at_location, find_location
from .agent import Agent, AgentState
from .memory import MemoryStore, auto_importance
from .llm import LLMProvider
from .dialogue import DialogueManager
from .dialogue_store import DialogueStore
from .embedding import EmbeddingProvider
from .relationship import RelationshipStore
from .trace import Trace
from .town_agent import TownAgent, TownState
from .tools import ToolRegistry
from .tools_full import make_registry_for
from .experience import ExperienceEvent, MemoryFormation
from .tasks import TaskStore
from .routine_planner import RoutinePlanner
from .candidates import build_routine_candidates
from .context import ContextService
from .cognition import (
    ActionExpectation, ActionOutcome, FactEvent, FactLedger, Goal,
)
from .interactions import InteractionEngine, ResourceStore
from .habits import HabitStore
from .emotion import apply_event as apply_emotion_event, decay as decay_emotion, profile_from_data
from .logistics import LogisticsEngine
from .decision import CognitivePlanner
from .request_state import (
    RequestState, capture_dialogue_request, capture_request_state,
    dialogue_stale_reason, stale_reason,
)
from .cognition_store import CognitionStore
from .interaction_store import InteractionRecord, InteractionStore
from .procedures import ProcedureStore
from .world_pack import DEFAULT_WORLD


AGENT_DEFS = DEFAULT_WORLD.agents


class SimulationEngine:
    """The core simulation engine integrating all Phase 1 components.

    Manages the simulation loop: time advancement, TownAgent polling, agent state
    machines (IDLE/MOVING/ACTING/SPEAKING/LISTENING), colocation relationship
    tracking, group-chat dialogue orchestration, and periodic reflection.

    The engine *is* the engine_ref — it is passed directly to agents, tools, and
    dialogue so they can reach every subsystem via dot-notation attribute access.
    """

    def __init__(self):
        # ── Core infrastructure ──
        self.llm = LLMProvider()
        self.memory = MemoryStore()
        self.relationship_store = RelationshipStore()
        self.trace = Trace()
        self.world_pack = DEFAULT_WORLD
        self.world_id = self.world_pack.id
        self.world_name = self.world_pack.name
        self.town_agent = TownAgent(self.world_pack)
        self.dialogue_store = DialogueStore(self.memory.db_path)
        self.cognition_store = CognitionStore(self.memory.db_path)
        self.procedure_store = ProcedureStore(self.memory.db_path)
        self.interaction_store = InteractionStore(self.memory.db_path)
        self._interactions: dict[str, InteractionRecord] = {}

        self.dialogue = DialogueManager(
            self.llm, self.memory, self.relationship_store,
            self._make_tool_registry, self.dialogue_store,
        )

        # ── Embedding provider for perception-association ──
        self.embedding_provider = EmbeddingProvider()
        self.procedure_store.provider = self.embedding_provider
        self.memory.set_embedding_provider(self.embedding_provider)
        self.memory_formation = MemoryFormation(self.memory, self.llm, self.trace)
        self.fact_ledger = FactLedger()
        self.dialogue.fact_ledger = self.fact_ledger
        self.resources = ResourceStore(self.world_pack)
        self.resources.reset()
        self.interactions = InteractionEngine(self.resources)
        self.context = ContextService(self.world_pack, self.resources)
        self.habits = HabitStore()
        self.planner = CognitivePlanner(self.llm)
        self.tasks = TaskStore()
        self.routine_planner = RoutinePlanner(self.tasks)
        self.logistics = LogisticsEngine(
            self.world_pack, self.resources, self.resources.scene, self.tasks,
        )
        self.logistics.reset()

        # ── Simulation state ─
        self.agents: list[Agent] = []
        self.day = 1
        self.hour = config.SIM_START_HOUR
        self.minute = 0
        self.running = False
        self.speed = 1  # ticks per second
        self._tick_count = 0
        self.state_version = 0
        self._last_reflection_timestamp = self.get_sim_timestamp()
        self._last_reflection_day = 0
        self._day_just_changed = False
        self._simulation_error: dict | None = None

        # The interactive run loop never waits on a remote completion.  Direct
        # tick() calls retain the synchronous behavior used by deterministic
        # tests and offline tooling.
        self._realtime_llm = False
        self._task_generation = 0
        self._planning_tasks: dict[str, tuple[int, asyncio.Task]] = {}
        self._daily_plan_tasks: dict[str, tuple[int, asyncio.Task]] = {}
        self._routine_choice_tasks: dict[str, tuple[int, asyncio.Task, dict]] = {}
        self._dialogue_turn_tasks: dict[str, tuple[int, asyncio.Task]] = {}
        self._planning_request_states: dict[str, RequestState] = {}
        self._daily_plan_request_states: dict[str, RequestState] = {}
        self._routine_choice_request_states: dict[str, RequestState] = {}
        self._dialogue_request_states: dict[str, object] = {}
        self._background_tasks: set[asyncio.Task] = set()

        # ── Town-level state (populated each tick) ──
        self.weather: dict = self.world_pack.initial_weather()
        self.infrastructure: dict = self.town_agent.infrastructure.to_dict()
        self.town_events: list[dict] = []
        self.active_festival: str | None = None
        self.season: str = "summer"

        # ── Pub/sub for WebSocket broadcast ──
        self._subscribers: list[asyncio.Queue] = []
        self._last_events: list[dict] = []
        self._event_sequence: int = 0

        # ── Dialogue intent queue (collected during agent processing) ──
        self._dialogue_intents: list[dict] = []
        self._encountered_pairs: set[tuple[str, str]] = set()
        self._active_encounter_pairs: set[tuple[str, str]] = set()
        self._active_colocation_groups: dict[str, tuple[str, ...]] = {}

    # ═══════════════════════════════════════════════════════════════
    # Engine as engine_ref
    #
    # Tools and agents access the engine via dot notation:
    #   engine.agents, engine.memory, engine.relationship_store,
    #   engine.hour, engine.minute, engine.weather, engine.town_agent,
    #   engine.trace, engine.town_events
    # ═══════════════════════════════════════════════════════════════

    def get_sim_time_str(self) -> str:
        return f"第{self.day}天 {self.hour:02d}:{self.minute:02d}"

    def get_sim_timestamp(self) -> int:
        return (self.day - 1) * 1440 + self.hour * 60 + self.minute

    def _make_tool_registry(self, agent_id: str) -> ToolRegistry:
        """Factory passed to DialogueManager for per-speaker tool registries.

        Passes `self` as engine_ref since tools_full handlers use dot-notation
        attribute access (engine.agents, engine.memory, etc.).
        """
        return make_registry_for(agent_id, self)

    # ═══════════════════════════════════════════════════════════════
    # Initialization
    # ═══════════════════════════════════════════════════════════════

    async def init(self):
        """Initialize all subsystems: DBs, agents, town systems."""
        # All SQLite-backed stores for one simulation share one database.
        # Tests can override memory.db_path before init() to isolate a run.
        storage_path = self.memory.db_path
        self.relationship_store.db_path = storage_path
        self.trace.db_path = storage_path
        self.dialogue_store.db_path = storage_path
        self.cognition_store.db_path = storage_path
        self.procedure_store.db_path = storage_path
        self.interaction_store.db_path = storage_path
        await self.memory.init_db()
        await self.relationship_store.init_db()
        await self.trace.init_db()
        await self.dialogue_store.init_db()
        await self.cognition_store.init_db()
        migrated = await self.procedure_store.init_db()
        for key in migrated:
            self._run_in_background(
                lambda key=key: self.procedure_store.index(key),
                "procedure_index_failed", self.get_sim_time_str(),
            )
        await self.interaction_store.init_db()
        self._init_agents()
        for agent in self.agents:
            await self.cognition_store.load_agent(agent, self.habits)

    def _init_agents(self):
        """Create Agent instances from definitions."""
        self.agents = []
        for ad in AGENT_DEFS:
            initial_needs = dict(ad.get("initial_needs", {}))
            needs = {
                key: max(0.0, min(100.0, float(initial_needs.get(key, default))))
                for key, default in {"energy": 75, "hunger": 25, "social": 55}.items()
            }
            state = AgentState(
                id=ad["id"], name=ad["name"], age=ad["age"],
                occupation=ad["occupation"], personality=ad["personality"],
                background=ad["background"], x=ad["x"], y=ad["y"],
                current_location=ad["location"], emoji=ad.get("emoji", "😐"),
                needs=needs,
            )
            agent = Agent(
                state, ad["schedule"], self.llm, self.memory,
                routine_blocks=self.world_pack.routine_blocks_for(ad["id"]),
                emotion_profile=profile_from_data(ad.get("emotion_profile")),
            )
            agent.duties = [
                duty.to_dict() for duty in self.resources.scene.duties_for(agent.id)
            ]
            for item in agent.schedule:
                agent.mental_state.add_goal(Goal(
                    description=item.label,
                    source=f"schedule:{item.hour:02d}{item.minute:02d}", created_at=0,
                    importance=round(0.45 + item.responsibility * 0.4, 2),
                    urgency=0.2, responsibility=item.responsibility,
                    flexibility=item.flexibility, deadline=item.start_minutes,
                    affected_agents=list(item.affected_people),
                    success_conditions=[
                        {"type": "at_location", "location": item.location},
                        {"type": "activity_completed", "description": item.activity or item.label},
                    ], status="proposed",
                ))
            self.agents.append(agent)

    # ═══════════════════════════════════════════════════════════════
    # State broadcast (pub/sub for WebSocket)
    # ═══════════════════════════════════════════════════════════════

    def get_state(self) -> dict:
        return {
            "world": {"id": self.world_id, "name": self.world_name},
            "map": {"width": self.world_pack.data["map"]["width"], "height": self.world_pack.data["map"]["height"]},
            "time": {"day": self.day, "hour": self.hour, "minute": self.minute},
            "speed": self.speed,
            "running": self.running,
            "stateVersion": self.state_version,
            "agents": [a.state.to_dict() for a in self.agents],
            "dailyPlans": [agent.daily_plan.to_dict() for agent in self.agents if agent.daily_plan],
            "locations": [
                {"id": loc.id, "name": loc.name, "type": loc.type,
                 "x": loc.x, "y": loc.y, "width": loc.width, "height": loc.height,
                 "color": loc.color, "emoji": loc.emoji}
                for loc in LOCATIONS
            ],
            "weather": self.weather,
            "infrastructure": self.infrastructure,
            "townEvents": self.town_events,
            "environment": self.resources.world_state([agent.id for agent in self.agents]),
            "scene": self.resources.scene_state(),
            "tasks": self.tasks.to_dict(),
            "habits": {agent.id: [habit.to_dict() for habit in self.habits.for_agent(agent.id)] for agent in self.agents},
            "logistics": self.logistics.state(),
            "recentEvents": self._last_events,
            "season": getattr(self, "season", "summer"),
            "festival": getattr(self, "active_festival", None),
            "health": {
                "simulation": "degraded" if self._simulation_error else "healthy",
                "simulationError": self._simulation_error,
                "llm": (
                    "fallback" if self.llm.fallback else
                    "degraded" if self._simulation_error else
                    "busy" if self._planning_tasks or self._daily_plan_tasks or self._routine_choice_tasks or self._dialogue_turn_tasks else
                    "connected"
                ),
                "embedding": "fallback" if self.embedding_provider.fallback else "connected",
                "dialogue": "healthy",
                "memory": "healthy",
                "reflection": "fallback-disabled" if self.llm.fallback else "healthy",
            },
        }

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        await q.put(self.get_state())
        return q

    def unsubscribe(self, q: asyncio.Queue):
        if q in self._subscribers:
            self._subscribers.remove(q)

    async def _complete_plan_step(self, agent, step_id):
        experience = agent.mental_state.complete_plan_step(step_id, self.get_sim_timestamp())
        if experience:
            key = await self.procedure_store.learn(agent.id, experience)
            self._run_in_background(
                lambda: self.procedure_store.index(key), "procedure_index_failed", self.get_sim_time_str(),
            )

    async def _block_plan(self, agent, plan, fact_id, reason):
        if plan.status != "blocked" and plan.experience_id:
            await self.procedure_store.fail(
                agent.id, plan.experience_id, reason, self.get_sim_timestamp(),
            )
        plan.block_current_step(fact_id, reason)

    async def _persist_cognition(self) -> None:
        for agent in self.agents:
            await self.cognition_store.save_agent(agent, self.habits)

    async def broadcast(self, events: list[dict] | None = None):
        if events:
            # Assign a monotonic sequence because one tick may emit events
            # created before and after the clock advances.
            for event in events:
                self._event_sequence += 1
                event.setdefault("sequence", self._event_sequence)
            self._last_events = (self._last_events + events)[-60:]
            self._last_events.sort(key=lambda event: (int(event.get("simTimestamp", 0)), int(event.get("sequence", 0))) )
        state = self.get_state()
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(state)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

    # ═══════════════════════════════════════════════════════════════
    # Simulation control
    # ═══════════════════════════════════════════════════════════════

    async def start(self):
        self._realtime_llm = True
        self.running = True
        self._tick_count = 0
        self._last_reflection_timestamp = self.get_sim_timestamp()
        asyncio.create_task(self._run_loop())

    async def _run_loop(self):
        while self.running:
            try:
                await self.tick()
            except Exception as e:
                import traceback
                print(f"[ENGINE] tick crashed: {e}")
                traceback.print_exc()
                self.running = False
                break
            self._tick_count += 1
            await asyncio.sleep(config.REALTIME_TICK_SECONDS / max(self.speed, 0.1))

    def pause(self):
        self.running = False

    async def shutdown(self) -> None:
        """Stop deferred work before closing the shared provider connection."""
        self.running = False
        tasks = self._cancel_deferred_tasks()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._persist_cognition()
        await self.llm.close()

    async def reset(self):
        self.running = False
        self._realtime_llm = False
        tasks = self._cancel_deferred_tasks()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.day = 1
        self.hour = config.SIM_START_HOUR
        self.minute = 0
        self._tick_count = 0
        self.state_version = 0
        self._last_reflection_timestamp = self.get_sim_timestamp()
        self._last_reflection_day = 0
        self._dialogue_intents.clear()
        self._interactions.clear()
        self._encountered_pairs.clear()
        self._active_encounter_pairs.clear()
        self._active_colocation_groups.clear()
        self._day_just_changed = False
        self._simulation_error = None
        self.dialogue._active_dialogues.clear()
        self.memory_formation._episode_buffers.clear()
        self.tasks.clear()
        self.habits.clear()
        self.fact_ledger.clear()
        self.resources.reset()
        self.logistics.reset()
        self.town_agent = TownAgent(self.world_pack)
        self.weather = self.world_pack.initial_weather()
        self.infrastructure = self.town_agent.infrastructure.to_dict()
        self.town_events = []
        self.active_festival = None
        self.season = "summer"
        self._last_events = []
        self._event_sequence = 0
        await self.memory.clear()
        await self.trace.clear()
        await self.relationship_store.clear()
        await self.dialogue_store.clear()
        await self.cognition_store.clear()
        await self.procedure_store.clear()
        await self.interaction_store.clear()
        self._init_agents()
        await self.broadcast()

    def set_speed(self, speed: float):
        self.speed = max(0.1, min(speed, 10.0))

    def _cancel_deferred_tasks(self) -> list[asyncio.Task]:
        """Invalidate work and return tasks for shutdown/reset to drain."""
        self._task_generation += 1
        tasks = [
            *(task for _, task in self._planning_tasks.values()),
            *(task for _, task in self._daily_plan_tasks.values()),
            *(task for _, task, _ in self._routine_choice_tasks.values()),
            *(task for _, task in self._dialogue_turn_tasks.values()),
            *self._background_tasks,
        ]
        for task in tasks:
            if task.done() and not task.cancelled():
                task.exception()
            task.cancel()
        self._planning_tasks.clear()
        self._daily_plan_tasks.clear()
        self._routine_choice_tasks.clear()
        self._dialogue_turn_tasks.clear()
        self._planning_request_states.clear()
        self._daily_plan_request_states.clear()
        self._routine_choice_request_states.clear()
        self._dialogue_request_states.clear()
        self._background_tasks.clear()
        return tasks

    def _run_in_background(self, work, label: str, sim_time: str) -> None:
        """Run non-critical enrichment without making the simulation wait."""
        generation = self._task_generation

        async def runner():
            try:
                await work()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if generation == self._task_generation:
                    await self.trace.log(sim_time, "system", "llm", label, str(exc)[:200])

        task = asyncio.create_task(runner(), name=f"town-{label}")
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # ═══════════════════════════════════════════════════════════════
    # Main tick loop
    # ═══════════════════════════════════════════════════════════════

    async def _plan_in_background(self, agent: Agent, sim_time_str: str, reason: str):
        context = await agent.build_planning_context(
            sim_time_str, self, task_type="intention_proposal",
        )
        return await self.planner.create_life_plan(agent, self, context, reason)

    async def _daily_plan_in_background(self, agent: Agent, sim_time_str: str,
                                        reason: str = ""):
        context = await agent.build_planning_context(
            sim_time_str, self, task_type="daily_plan",
        )
        return await self.planner.create_daily_plan(agent, self, context, reason)

    async def _collect_daily_plans_realtime(self, sim_time_str: str) -> None:
        """Commit ready coarse plans and request at most one per agent/day."""
        ready_ids = [agent_id for agent_id, (_, task) in self._daily_plan_tasks.items() if task.done()]
        for agent_id in ready_ids:
            generation, task = self._daily_plan_tasks.pop(agent_id)
            request = self._daily_plan_request_states.pop(agent_id, None)
            if generation != self._task_generation:
                continue
            agent = self._find_agent_by_id(agent_id)
            if not agent:
                continue
            reason = stale_reason(self, agent, request) if request else ""
            if reason:
                await self.trace.log(
                    sim_time_str, agent.id, "llm", "daily_plan_stale", reason,
                )
                continue
            try:
                plan = task.result()
                agent.daily_plan = plan
                await self.trace.log(sim_time_str, agent.id, "daily_plan", "created", plan.focus[:200])
            except asyncio.CancelledError:
                continue
            except Exception as exc:
                await self.trace.log(sim_time_str, agent.id, "daily_plan", "failed", str(exc)[:200])

        if self.llm.fallback:
            return
        for agent in self.agents:
            plan = agent.daily_plan
            needs_plan = plan is None or plan.day != self.day
            if needs_plan and agent.id not in self._daily_plan_tasks:
                reason = plan.replan_reason if plan else ""
                task = asyncio.create_task(
                    self._daily_plan_in_background(agent, sim_time_str, reason),
                    name=f"town-daily-plan-{agent.id}",
                )
                self._daily_plan_tasks[agent.id] = (self._task_generation, task)
                self._daily_plan_request_states[agent.id] = capture_request_state(
                    self, agent, "daily_plan",
                )

    async def _select_routine_candidate_in_background(self, agent: Agent, candidates):
        return await self.planner.select_routine_candidate(agent, self, candidates)

    async def _collect_routine_choices_realtime(self, sim_time_str: str) -> dict[str, dict]:
        """Use ready LLM choices, preserving deterministic fallback on failure."""
        decisions: dict[str, dict] = {}
        ready_ids = [agent_id for agent_id, (_, task, _) in self._routine_choice_tasks.items() if task.done()]
        for agent_id in ready_ids:
            generation, task, fallback_action = self._routine_choice_tasks.pop(agent_id)
            request = self._routine_choice_request_states.pop(agent_id, None)
            if generation != self._task_generation:
                continue
            agent = self._find_agent_by_id(agent_id)
            if not agent or agent.state.status != "IDLE":
                continue
            reason = stale_reason(self, agent, request) if request else ""
            if reason:
                decisions[agent_id] = fallback_action
                await self.trace.log(
                    sim_time_str, agent.id, "llm", "routine_choice_stale", reason,
                )
                continue
            try:
                decisions[agent_id] = task.result()
                await self.trace.log(sim_time_str, agent_id, "routine", "llm_candidate_selected", str(decisions[agent_id])[:200])
            except asyncio.CancelledError:
                continue
            except Exception as exc:
                decisions[agent_id] = fallback_action
                await self.trace.log(sim_time_str, agent_id, "routine", "llm_candidate_fallback", str(exc)[:200])
        return decisions

    async def _collect_realtime_decisions(self, sim_time_str: str) -> tuple[dict[str, dict], list[dict]]:
        """Commit completed plans and enqueue missing ones without awaiting LLM I/O."""
        decisions: dict[str, dict] = {}
        events: list[dict] = []
        await self._collect_daily_plans_realtime(sim_time_str)
        decisions.update(await self._collect_routine_choices_realtime(sim_time_str))
        ready_ids = [agent_id for agent_id, (_, task) in self._planning_tasks.items() if task.done()]
        for agent_id in ready_ids:
            generation, task = self._planning_tasks.pop(agent_id)
            request = self._planning_request_states.pop(agent_id, None)
            if generation != self._task_generation:
                continue
            agent = self._find_agent_by_id(agent_id)
            if not agent:
                continue
            reason = stale_reason(self, agent, request) if request else ""
            if reason:
                await self.trace.log(
                    sim_time_str, agent.id, "llm", "life_plan_stale", reason,
                )
                continue
            try:
                plan = task.result()
                agent.mental_state.life_plan = plan
                await self.trace.log(sim_time_str, agent.id, "plan", "life_plan_created", plan.focus[:200])
                decisions[agent.id] = self.planner.next_plan_action(agent, self)
                await self.trace.log(sim_time_str, agent.id, "plan", "step_selected", str(decisions[agent.id])[:200])
                self._simulation_error = None
            except asyncio.CancelledError:
                continue
            except Exception as exc:
                self._simulation_error = {
                    "type": "decision_deferred",
                    "simTime": sim_time_str,
                    "agentIds": [agent.id],
                    "message": "角色的规划请求失败；世界仍在继续运行",
                }
                await self.trace.log(sim_time_str, agent.id, "llm", "plan_failed", str(exc)[:200])
                events.append(self._make_event({
                    "type": "decision_deferred", "agentIds": [agent.id], "location": "",
                    "content": self._simulation_error["message"], "cause": "llm_decision_failure",
                }))

        dialogue_participants = self.dialogue.participant_ids()
        now = self.get_sim_timestamp()
        for agent in self.agents:
            if agent.id in dialogue_participants or agent.state.status != "IDLE" or agent.id in decisions:
                continue
            agent.sync_schedule_goals(self.day, self.hour, self.minute)
            current_slot = agent._get_schedule_item(self.hour, self.minute)
            if current_slot and current_slot.is_flexible_slot and not self.llm.fallback:
                candidates = build_routine_candidates(agent, self)
                if candidates:
                    if agent.id not in self._routine_choice_tasks:
                        task = asyncio.create_task(
                            self._select_routine_candidate_in_background(agent, candidates),
                            name=f"town-routine-choice-{agent.id}",
                        )
                        self._routine_choice_tasks[agent.id] = (
                            self._task_generation, task, dict(candidates[0].action),
                        )
                        self._routine_choice_request_states[agent.id] = capture_request_state(
                            self, agent, "routine_selection",
                        )
                        agent.state.current_action = "在几个安排中权衡"
                    continue
            routine_action = self.routine_planner.next_action(agent, self)
            if routine_action:
                decisions[agent.id] = routine_action
                await self.trace.log(
                    sim_time_str, agent.id, "routine", "task_selected",
                    str(routine_action)[:200],
                )
                continue

            plan = agent.mental_state.life_plan
            if plan and plan.status == "active" and plan.current_step:
                try:
                    decisions[agent.id] = self.planner.next_plan_action(agent, self)
                    await self.trace.log(
                        sim_time_str, agent.id, "plan", "step_selected",
                        str(decisions[agent.id])[:200],
                    )
                    continue
                except Exception as exc:
                    await self._block_plan(agent, plan, "", str(exc))

            # Live planning runs in the background so an observed failure can
            # produce a new branch without stalling the simulation clock.
            if not self.llm.fallback and agent.id not in self._planning_tasks:
                reason = plan.replan_reason if plan and plan.status == "blocked" else ""
                task = asyncio.create_task(
                    self._plan_in_background(agent, sim_time_str, reason),
                    name=f"town-plan-{agent.id}",
                )
                self._planning_tasks[agent.id] = (self._task_generation, task)
                self._planning_request_states[agent.id] = capture_request_state(
                    self, agent, "life_plan",
                )
                agent.state.current_action = "根据当前情况考虑下一步"
            elif agent.id not in self._planning_tasks:
                agent.state.current_action = "等待下一项任务"

        return decisions, events

    async def _tick_realtime(self):
        """Interactive tick whose wall-clock duration is independent of LLM latency."""
        sim_time_str = self.get_sim_time_str()
        all_events: list[dict] = []

        for agent in self.agents:
            if agent.state.status == "MOVING":
                all_events.extend(await self._process_moving(agent, sim_time_str))
            elif agent.state.status == "ACTING":
                all_events.extend(await self._process_acting(agent, sim_time_str))

        decisions, decision_events = await self._collect_realtime_decisions(sim_time_str)
        all_events.extend(decision_events)

        self._advance_time()
        if self._day_just_changed:
            self._day_just_changed = False
            await self._daily_maintenance()
        sim_time_str = self.get_sim_time_str()
        self._tick_town_systems()
        all_events.extend(self._advance_logistics())
        for agent in self.agents:
            self.town_agent.advance_agent_needs(agent)
            decay_emotion(agent)
        await self._update_colocation(sim_time_str)
        if self.hour >= 21 and self._last_reflection_day != self.day:
            self._run_in_background(lambda: self._run_reflections(sim_time_str), "reflection_failed", sim_time_str)
            self._last_reflection_timestamp = self.get_sim_timestamp()
            self._last_reflection_day = self.day

        self._dialogue_intents = []
        for agent in self.agents:
            decision = decisions.get(agent.id)
            if decision and agent.state.status == "IDLE":
                all_events.extend(await self._execute_decision(agent, decision, sim_time_str))

        self._encountered_pairs = set(self._active_encounter_pairs)
        all_events.extend(await self._advance_dialogues_realtime(sim_time_str))
        all_events.extend(await self._detect_encounters(sim_time_str))
        all_events.extend(await self._run_group_dialogues(sim_time_str))
        await self._persist_cognition()
        await self.broadcast(all_events)

    async def tick(self):
        """One simulation tick.

        Expiration is a state transition owned by the engine. Context and API
        reads remain side-effect free.

        1. Process MOVING/ACTING agents and their consequences
        2. Resolve scheduled commitments and urgent needs without an LLM
        3. Collect only remaining IDLE agents; wait for all LLM decisions
        4. Advance time, town systems, colocation, and needs
        5. Execute all validated intentions as one tick commit
        5. Dialogue, encounters, broadcast
        """
        now = self.get_sim_timestamp()
        self.state_version += 1
        for agent in self.agents:
            agent.mental_state.expire_intentions(now)
        if self._realtime_llm:
            await self._tick_realtime()
            return

        sim_time_str = self.get_sim_time_str()
        all_events: list[dict] = []

        # ── Phase 1: Process MOVING/ACTING agents ──
        for agent in self.agents:
            if agent.state.status == "MOVING":
                events = await self._process_moving(agent, sim_time_str)
                all_events.extend(events)
            elif agent.state.status == "ACTING":
                events = await self._process_acting(agent, sim_time_str)
                all_events.extend(events)

        # ── Phase 2: Continue life plans; plan only at explicit boundaries ──
        decisions: dict[str, dict] = {}
        planning_agents: list[Agent] = []
        dialogue_participants = self.dialogue.participant_ids()
        now = self.get_sim_timestamp()
        for agent in self.agents:
            if agent.id in dialogue_participants or agent.state.status != "IDLE":
                continue
            agent.sync_schedule_goals(self.day, self.hour, self.minute)
            plan = agent.mental_state.life_plan
            needs_plan = (
                plan is None or plan.status in {"completed", "blocked", "abandoned"}
                or (plan.review_at <= now and plan.current_step is None)
            )
            if plan and plan.status == "active" and plan.current_step and plan.current_step.status == "executing":
                await self._block_plan(agent, plan, "", "执行状态与角色空闲状态不一致")
                needs_plan = True
            if needs_plan:
                planning_agents.append(agent)
                continue
            try:
                decisions[agent.id] = self.planner.next_plan_action(agent, self)
                await self.trace.log(
                    sim_time_str, agent.id, "plan", "step_selected",
                    str(decisions[agent.id])[:200],
                )
            except Exception as exc:
                await self._block_plan(agent, plan, "", str(exc))
                planning_agents.append(agent)

        # A plan is retained across ticks. Retry only plan creation failures;
        # successful plans and steps are never regenerated in the same tick.
        decision_attempt = 0
        while planning_agents and decision_attempt < 2:
            decision_attempt += 1
            contexts = await asyncio.gather(*[
                agent.build_planning_context(
                    sim_time_str, self, task_type="intention_proposal",
                )
                for agent in planning_agents
            ], return_exceptions=True)
            tasks = []
            task_agents = []
            failed_context_agents = []
            for agent, context in zip(planning_agents, contexts):
                if isinstance(context, Exception):
                    failed_context_agents.append(agent)
                    continue
                previous = agent.mental_state.life_plan
                reason = previous.replan_reason if previous and previous.status == "blocked" else ""
                tasks.append(self.planner.create_life_plan(agent, self, context, reason))
                task_agents.append(agent)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            still_planning = list(failed_context_agents)
            for agent, result in zip(task_agents, results):
                if isinstance(result, Exception):
                    await self.trace.log(sim_time_str, agent.id, "llm", "retry", str(result)[:200])
                    still_planning.append(agent)
                    continue
                agent.mental_state.life_plan = result
                await self.trace.log(sim_time_str, agent.id, "plan", "life_plan_created", result.focus[:200])
                try:
                    decisions[agent.id] = self.planner.next_plan_action(agent, self)
                except Exception as exc:
                    await self._block_plan(agent, result, "", str(exc))
                    still_planning.append(agent)
            planning_agents = still_planning
            if planning_agents and not self.llm.fallback:
                await asyncio.sleep(0.5)
        idle_agents = planning_agents
        if idle_agents:
            failed_ids = [agent.id for agent in idle_agents]
            self._simulation_error = {
                "type": "decision_deferred",
                "simTime": sim_time_str,
                "agentIds": failed_ids,
                "message": "部分角色本tick未取得有效结构化决策；其余角色和世界继续运行",
            }
            for agent in idle_agents:
                await self.trace.log(
                    sim_time_str, agent.id, "warning", "decision_deferred",
                    self._simulation_error["message"],
                )
            all_events.append(self._make_event({
                "type": "decision_deferred",
                "agentIds": failed_ids,
                "location": "",
                "content": self._simulation_error["message"],
                "cause": "llm_decision_failure",
            }))
        else:
            self._simulation_error = None
        # ── Phase 3: All decisions ready → advance time ──
        self._advance_time()
        if self._day_just_changed:
            self._day_just_changed = False
            await self._daily_maintenance()
        sim_time_str = self.get_sim_time_str()
        self._tick_town_systems()
        all_events.extend(self._advance_logistics())
        for agent in self.agents:
            self.town_agent.advance_agent_needs(agent)
            decay_emotion(agent)
        await self._update_colocation(sim_time_str)
        if self.hour >= 21 and self._last_reflection_day != self.day:
            await self._run_reflections(sim_time_str)
            self._last_reflection_timestamp = self.get_sim_timestamp()
            self._last_reflection_day = self.day

        # ── Phase 4: Execute all decisions ──
        self._dialogue_intents = []
        for agent in self.agents:
            decision = decisions.get(agent.id)
            if decision and agent.state.status == "IDLE":
                events = await self._execute_decision(agent, decision, sim_time_str)
                all_events.extend(events)

        # ── Phase 5: Advance existing dialogues, start new ones, encounters ──
        self._encountered_pairs = set(self._active_encounter_pairs)
        existing_dialogue_events = await self._advance_dialogues(sim_time_str)
        all_events.extend(existing_dialogue_events)
        self._queue_colocated_dialogues()
        encounter_events = await self._detect_encounters(sim_time_str)
        all_events.extend(encounter_events)
        dialogue_events = await self._run_group_dialogues(sim_time_str)
        all_events.extend(dialogue_events)
        await self._persist_cognition()
        await self.broadcast(all_events)
    # ═══════════════════════════════════════════════════════════════
    # Tick sub-phases
    # ═══════════════════════════════════════════════════════════════
    def _advance_time(self):
        """Advance sim clock by TICK_INTERVAL_MINUTES."""
        self.minute += config.TICK_INTERVAL_MINUTES
        if self.minute >= 60:
            self.hour += self.minute // 60
            self.minute %= 60
        if self.hour >= config.SIM_END_HOUR:
            self.day += 1
            self.hour = config.SIM_START_HOUR
            self._day_just_changed = True


    def _schedule_action(self, agent: Agent) -> str:
        """Return a schedule-appropriate action text."""
        s = agent.get_schedule_for_time(self.hour, self.minute)
        for line in s.split('\n'):
            if line.startswith('当前应该:'):
                return line.split('当前应该: ')[1].split('（')[0].strip()
        return '做自己的事'
    def _tick_town_systems(self):
        """Query TownAgent for weather, infrastructure, random events, and seasons."""
        try:
            agent_locations = {a.state.id: a.state.current_location for a in self.agents}
            town_state: TownState = self.town_agent.tick(self.hour, self.day, agent_locations)
            self.weather = town_state.weather
            self.infrastructure = town_state.infrastructure
            self.town_events = town_state.events
        except Exception as exc:
            print(f"[TOWN] world-system update failed: {exc}", flush=True)

    def _advance_logistics(self) -> list[dict]:
        """Advance deterministic orders and expose them as public events."""
        return [
            self._make_event({
                "type": item["type"], "agentIds": [], "location": "",
                "content": item["type"], "cause": "logistics",
            })
            for item in self.logistics.advance(self.get_sim_timestamp())
        ]

    async def _update_colocation(self, sim_time_str: str):
        """Log a colocated group only when its membership changes."""
        current_groups: dict[str, tuple[str, ...]] = {}
        for agent in self.agents:
            loc = agent.state.current_location
            if loc == "in_transit":
                continue
            current_groups.setdefault(loc, tuple())
            current_groups[loc] = tuple(sorted((*current_groups[loc], agent.id)))
        current_groups = {
            loc: group for loc, group in current_groups.items() if len(group) >= 2
        }
        for loc, group in current_groups.items():
            if self._active_colocation_groups.get(loc) == group:
                continue
            names = "、".join(
                agent.name for agent in self.agents if agent.id in group
            )
            await self.trace.log(
                sim_time_str, "system", "state", "colocation",
                f"{names}同在{location_name(loc)}",
            )
        for loc, previous in self._active_colocation_groups.items():
            if loc not in current_groups:
                names = "、".join(
                    agent.name for agent in self.agents if agent.id in previous
                )
                await self.trace.log(
                    sim_time_str, "system", "state", "colocation_ended",
                    f"{names}不再同处{location_name(loc)}",
                )
        for group in current_groups.values():
            await self.relationship_store.record_colocation(list(group))
        self._active_colocation_groups = current_groups

    def _queue_colocated_dialogues(self) -> None:
        """Allow one idle pair sharing a social space to start a conversation."""
        for location, group in self._active_colocation_groups.items():
            participants = [self._find_agent_by_id(agent_id) for agent_id in group]
            idle = [agent for agent in participants if agent and agent.state.status == "IDLE"]
            if len(idle) < 2:
                continue
            self._queue_encounter_dialogue(idle[0], idle[1])

    async def _start_interaction(self, agent: Agent, action: dict,
                                 expected_effects: list[dict]) -> InteractionRecord:
        interaction_id = str(action.get("interaction_id", ""))
        record = InteractionRecord(
            id=interaction_id,
            sim_timestamp=self.get_sim_timestamp(),
            state_version=self.state_version,
            initiator_id=agent.id,
            target_type="agent" if action.get("target") else "resource" if action.get("resource_id") else "location" if action.get("location") else "",
            target_id=str(action.get("target") or action.get("resource_id") or action.get("location") or ""),
            interaction_type=str(action.get("interaction_type") or action.get("action") or ""),
            intention_id=str(action.get("intention_id", "")),
            status="executing",
            payload=dict(action),
            expected_effects=list(expected_effects),
        )
        self._interactions[record.id] = record
        await self.interaction_store.upsert(record)
        return record

    async def _finish_interaction(self, interaction_id: str, status: str,
                                  effects: list[dict] | None = None,
                                  fact_id: str = "") -> None:
        record = self._interactions.get(interaction_id)
        if not record:
            record = await self.interaction_store.get(interaction_id)
        if not record:
            return
        record.status = status
        if effects is not None:
            record.actual_effects = list(effects)
        if fact_id and fact_id not in record.fact_ids:
            record.fact_ids.append(fact_id)
        self._interactions[record.id] = record
        await self.interaction_store.upsert(record)

    def _record_fact(self, agent: Agent, fact_type: str, location: str,
                     details: dict, participants: list[str] | None = None,
                     source_ids: list[str] | None = None,
                     interaction_id: str = "") -> FactEvent:
        fact = self.fact_ledger.add(FactEvent(
            sim_time=self.get_sim_time_str(),
            sim_timestamp=self.get_sim_timestamp(),
            type=fact_type,
            agent_id=agent.id,
            location=location,
            details=dict(details),
            participants=list(participants or []),
            source_ids=list(source_ids or []),
            interaction_id=interaction_id,
        ))
        agent.mental_state.add_fact(fact, goal=agent.mental_state.active_goal)
        return fact

    async def _form_experience(self, agent: Agent, event: ExperienceEvent):
        memory_emotion = apply_emotion_event(agent, event)
        if memory_emotion:
            event.facts["emotion_label"] = memory_emotion
        async def form():
            try:
                await self.memory_formation.process(agent, event, self)
            except Exception as exc:
                await self.trace.log(
                    self.get_sim_time_str(), agent.id, "error",
                    "memory_formation_failed", str(exc)[:300],
                )

        if self._realtime_llm:
            self._run_in_background(form, "memory_formation_failed", self.get_sim_time_str())
            return
        await form()

    async def _process_moving(self, agent: Agent, sim_time_str: str) -> list[dict]:
        path = agent._movement_path
        if not path:
            agent.state.status = "IDLE"
            return []
        steps = min(config.MOVE_SPEED, len(path))
        for _ in range(steps):
            next_cell = path.pop(0)
            agent.state.x, agent.state.y = next_cell
            # Report the actual current area while travelling instead of
            # retaining the building the agent left until arrival.
            loc = find_location(agent.state.x, agent.state.y)
            agent.state.current_location = loc.id if loc else "in_transit"
        if not path:
            agent._movement_path = None
            agent.state.status = "IDLE"
            # Set location from final position
            loc = find_location(agent.state.x, agent.state.y)
            if loc:
                agent.state.current_location = loc.id
            movement_action = agent._movement_action or {
                "interaction_type": "move", "action_id": "",
                "location": agent.state.current_location,
            }
            agent._movement_action = None
            arrival_fact = self._record_fact(
                agent, "movement_arrived", agent.state.current_location,
                {
                    "purpose": agent._movement_reason or "办事",
                    "weather": self.weather.get("condition", "") if isinstance(self.weather, dict) else str(self.weather),
                    "action_id": movement_action.get("action_id", ""),
                    "effects": [{"type": "location_change", "location": agent.state.current_location}],
                },
                interaction_id=str(movement_action.get("interaction_id", "")),
            )
            agent.mental_state.record_outcome(ActionOutcome(
                action_id=str(movement_action.get("action_id", "")),
                actual_effects=[{"type": "location_change", "location": agent.state.current_location}],
                duration=max(0, self.get_sim_timestamp() - int(movement_action.get("started_at", self.get_sim_timestamp()))),
                success=True, completed_at=self.get_sim_timestamp(),
            ))
            await self._complete_plan_step(
                agent, str(movement_action.get("plan_step_id", "")),
            )
            intention = agent.mental_state.relevant_intention(movement_action)
            if intention:
                agent.mental_state.transition_intention(
                    intention.id, "completed", self.get_sim_timestamp(), arrival_fact.id
                )
            events = [self._make_event({
                "type": "movement_arrived", "factId": arrival_fact.id,
                "interactionId": movement_action.get("interaction_id", ""),
                "agentIds": [agent.id],
                "location": agent.state.current_location,
                "content": f"{agent.name} · {location_name(agent.state.current_location)}",
                "cause": "movement_completed",
                "outcome": "已到达目的地",
            })]
            events.extend(await self._handle_arrival_perception(agent, sim_time_str))
            pending_commitment = (agent._pending_action or {}).get("commitment") or {}
            deadline = pending_commitment.get("deadline_minutes")
            current_minutes = self.hour * 60 + self.minute
            lateness = max(0, current_minutes - deadline) if deadline is not None else 0
            weather_condition = self.weather.get("condition", "") if isinstance(self.weather, dict) else str(self.weather)
            weather_cost = float(self.weather.get("travel_cost", 0.0)) if isinstance(self.weather, dict) else 0.0
            weather_salience = float(self.weather.get("sensory_salience", 0.0)) if isinstance(self.weather, dict) else 0.0
            await self._form_experience(agent, ExperienceEvent(
                type="arrival",
                location=agent.state.current_location,
                actors=[agent.id],
                source=agent._movement_source or "movement",
                expected=lateness <= 5 and weather_cost < 0.6,
                emotional_valence=-0.25 if lateness > 5 else 0.0,
                emotional_intensity=0.35 if lateness > 5 else 0.1,
                sensory_salience=weather_salience,
                facts={
                    "purpose": agent._movement_reason or "办事",
                    "lateness": lateness,
                    "weather": weather_condition,
                    "travel_cost": weather_cost,
                },
            ))
            agent._movement_reason = ""
            agent._movement_source = ""
            # Execute pending intent
            pending_action = agent._pending_action
            if pending_action:
                pending = pending_action
                agent._pending_action = None
                if pending.get("action") == "complete_travel":
                    agent.complete_travel_commitment((pending.get("commitment") or {}).get("id"))
                else:
                    pending_events = await self._execute_decision(agent, pending, sim_time_str)
                    events.extend(pending_events)
            if not pending_action or pending_action.get("action") == "complete_travel":
                await self._finish_interaction(
                    str(movement_action.get("interaction_id", "")), "completed",
                    effects=[{"type": "location_change", "location": agent.state.current_location}],
                    fact_id=arrival_fact.id,
                )
            return events
        return []

    async def _process_acting(self, agent: Agent, sim_time_str: str) -> list[dict]:
        if agent._activity_ticks <= 0:
            agent.state.status = "IDLE"
            return []
        agent._activity_ticks -= 1
        if agent._activity_ticks <= 0:
            desc = agent._activity_desc
            source = agent._activity_source or "activity"
            started_at = agent._activity_started_at
            agent._activity_desc = ""
            agent._activity_source = ""
            agent._activity_started_at = None
            agent.state.status = "IDLE"
            agent.state.current_action = "待机中"
            commitment_id = agent.complete_activity_commitment()
            action = agent._activity_action or {"interaction_type": "wait", "content": desc}
            self.habits.record(agent.id, action, self.get_sim_timestamp(), success=True)
            task_id = str(action.get("task_id", ""))
            effects = self.interactions.apply_effects(
                agent, action, agent._activity_effects, self.get_sim_timestamp()
            )
            completed_goal_effects = [
                effect for effect in effects
                if effect.get("type") == "goal_progress" and effect.get("completed")
            ]
            for effect in completed_goal_effects:
                agent.complete_goal_commitment(str(effect.get("goal_id", "")), self.day)
            agent._activity_action = None
            agent._activity_effects = []
            if task_id:
                self.tasks.complete(task_id, self.get_sim_timestamp())
            needs = agent.state.needs.copy()
            elapsed = self.get_sim_timestamp() - started_at if started_at is not None else 0
            need_driven = source.startswith("need:")
            await self._form_experience(agent, ExperienceEvent(
                type="activity_outcome",
                location=agent.state.current_location,
                actors=[agent.id],
                source=source,
                expected=source == "commitment",
                emotional_valence=0.35 if need_driven else 0.05,
                emotional_intensity=0.35 if need_driven else 0.1,
                goal_relevance=0.9 if completed_goal_effects else 0.0,
                facts={
                    "activity": desc,
                    "duration": elapsed,
                    "reason": source.removeprefix("need:") if need_driven else "日常安排",
                    "outcome": "完成了当前目标" if completed_goal_effects else "缓解了当时迫切的需要" if need_driven else "",
                    "effects": effects,
                    "goal_id": str(action.get("goal_id", "")),
                    "action_id": str(action.get("action_id", "")),
                },
            ))
            observation_fact = None
            observation_effect = next(
                (effect for effect in effects if effect.get("type") == "observation"), None
            )
            if observation_effect:
                snapshot = dict(observation_effect.get("snapshot", {}))
                observation_fact = self._record_fact(
                    agent, "environment_observed", agent.state.current_location,
                    {
                        "target_id": snapshot.get("target_id", ""),
                        "signature": snapshot.get("signature", ""),
                        "entities": snapshot.get("entities", []),
                        "nearby_agents": snapshot.get("nearby_agents", []),
                        "available_processes": snapshot.get("available_processes", []),
                        "weather": snapshot.get("weather", {}),
                        "observed_at": observation_effect.get("observed_at", self.get_sim_timestamp()),
                    },
                )
            completed_fact = self._record_fact(
                agent, "activity_completed", agent.state.current_location,
                {
                    "activity": desc,
                    "duration": elapsed,
                    "needs_after": needs,
                    "effects": effects,
                    "action_id": action.get("action_id", ""),
                    "goal": agent.mental_state.active_goal,
                },
                source_ids=[observation_fact.id] if observation_fact else None,
            )
            await self._finish_interaction(
                str(action.get("interaction_id", "")), "completed",
                effects=effects, fact_id=completed_fact.id,
            )
            outcome = ActionOutcome(
                action_id=str(action.get("action_id", "")), actual_effects=effects,
                duration=elapsed, success=True, completed_at=self.get_sim_timestamp(),
            )
            agent.mental_state.record_outcome(outcome)
            await self._complete_plan_step(agent, str(action.get("plan_step_id", "")))
            intention = agent.mental_state.relevant_intention(action)
            if intention:
                agent.mental_state.transition_intention(
                    intention.id, "completed", self.get_sim_timestamp(), completed_fact.id
                )
            environment_changes = self._summarize_environment_effects(effects)
            return [self._make_event({
                "type": "activity_completed", "factId": completed_fact.id,
                "interactionId": action.get("interaction_id", ""),
                "agentIds": [agent.id],
                "location": agent.state.current_location,
                "content": f"{agent.name}完成了{desc}",
                "cause": "commitment" if commitment_id else "activity_duration_elapsed",
                "outcome": environment_changes or f"需求：精力{needs['energy']}，饥饿{needs['hunger']}，社交{needs['social']}",
            })]
        return []

    def _summarize_environment_effects(self, effects: list[dict]) -> str:
        changes = []
        for effect in effects:
            effect_type = effect.get("type")
            resource = self.resources.get(str(effect.get("resource_id", "")))
            entity = self.resources.get(str(effect.get("entity_id", "")))
            resource_name = resource.name if resource else "资源"
            entity_name = entity.name if entity else "设施"
            if effect_type == "resource_change":
                changes.append(
                    f"{resource_name} {effect.get('before', '?')}→{effect.get('after', '?')}"
                )
            elif effect_type == "resource_move":
                destination = str(effect.get("destination", ""))
                destination_name = "随身物品" if destination.startswith("agent:") else location_name(destination)
                changes.append(f"{resource_name}移至{destination_name}")
            elif effect_type == "resource_add":
                changes.append(
                    f"{effect.get('name', effect.get('kind', '资源'))}+{effect.get('quantity', 0)}"
                )
            elif effect_type == "entity_used":
                changes.append(f"使用{entity_name}")
            elif effect_type == "service_fulfilled":
                staff = self._find_agent_by_id(str(effect.get("staff_id", "")))
                changes.append(f"由{staff.name if staff else '工作人员'}提供服务")
            elif effect_type == "observation":
                snapshot = effect.get("snapshot", {})
                changes.append(
                    f"看到{len(snapshot.get('entities', []))}项实体、"
                    f"{len(snapshot.get('nearby_agents', []))}位附近角色、"
                    f"{len(snapshot.get('available_processes', []))}项可用过程"
                )
        return "；".join(changes)

    def _public_action_text(self, agent: Agent, content: str) -> str:
        """Translate structured IDs and remove audit-only assumptions from UI text."""
        text = str(content or "").strip()
        for resource in self.resources.resources.values():
            if resource.id in text:
                text = text.replace(resource.id, resource.name)
        for goal in agent.mental_state.goals:
            if goal.id in text:
                text = text.replace(goal.id, goal.description)
        text = re.sub(r"[（(][^）)]*(?:假设|goal_|resource_|device_|process_)[^）)]*[）)]", "", text)
        text = re.sub(r"\b(?:goal|resource|device|process)_[A-Za-z0-9_]+\b", "", text)
        text = re.sub(rf"^{re.escape(agent.name)}(?:正在|开始|继续)?", "", text).strip(" ，,。；;")
        text = re.sub(r"\s+", " ", text).strip(" ，,。；;")
        return text[:160]

    def _validate_decision(self, agent: Agent, decision: dict) -> dict:
        """Clean display text without inventing a substitute life action."""
        normalized = dict(decision or {})
        raw_content = str(normalized.get("content") or "").strip()
        content = self._public_action_text(agent, raw_content)
        if raw_content and content != raw_content:
            normalized["audit_content"] = raw_content
            normalized["content"] = content
        return normalized

    def _validated_mental_update(self, agent: Agent, update: dict | None) -> tuple[dict, set[str]]:
        """Keep belief provenance limited to facts visible to this agent now."""
        normalized = dict(update or {})
        raw_updates = normalized.get("belief_updates")
        if not isinstance(raw_updates, list):
            return normalized, set()
        now = self.get_sim_timestamp()
        valid_ids: set[str] = set()
        cleaned = []
        for raw in raw_updates:
            if isinstance(raw, str):
                raw = {"proposition": raw}
            if not isinstance(raw, dict) or not str(raw.get("proposition", "")).strip():
                continue
            source_ids = raw.get("source_fact_ids", [])
            source_ids = source_ids if isinstance(source_ids, list) else [source_ids]
            valid, _ = self.fact_ledger.validate_references(
                [str(item) for item in source_ids], agent.id, now,
            )
            valid_ids.update(valid)
            item = dict(raw)
            item["source_fact_ids"] = valid
            if not valid:
                item["status"] = "uncertain"
                item["confidence"] = min(float(item.get("confidence", 0.5)), 0.35)
            cleaned.append(item)
        normalized["belief_updates"] = cleaned
        return normalized, valid_ids

    async def _execute_decision(self, agent: Agent, decision: dict,
                                 sim_time_str: str) -> list[dict]:
        """Validate and execute a decided action. Returns events."""
        decision = self._validate_decision(agent, decision)
        mental_update, valid_fact_ids = self._validated_mental_update(
            agent, decision.get("mental_update"),
        )
        decision["mental_update"] = mental_update
        agent.mental_state.apply_update(
            mental_update, self.get_sim_timestamp(), valid_fact_ids=valid_fact_ids,
        )
        if decision.get("action") == "noop":
            return []
        decision.setdefault("interaction_id", f"interaction_{uuid.uuid4().hex[:12]}")
        feasibility = self.interactions.validate(agent, decision, self)
        if not feasibility.feasible:
            await self._start_interaction(agent, decision, [])
            self.habits.record(agent.id, decision, self.get_sim_timestamp(), success=False)
            task_id = str(decision.get("task_id", ""))
            if task_id:
                self.tasks.block(task_id, "; ".join(feasibility.reasons))
            interaction_type = str(
                decision.get("interaction_type") or decision.get("action") or ""
            )
            blocked = self._record_fact(
                agent,
                "service_unavailable" if interaction_type == "request_service" else "action_blocked",
                agent.state.current_location,
                {"action": decision, "reasons": feasibility.reasons},
                interaction_id=str(decision.get("interaction_id", "")),
            )
            intention = agent.mental_state.relevant_intention(decision)
            if intention:
                agent.mental_state.transition_intention(
                    intention.id, "blocked", self.get_sim_timestamp(), blocked.id
                )
            agent.mental_state.record_outcome(ActionOutcome(
                action_id=str(feasibility.resolved_action.get("action_id", "")),
                actual_effects=[], duration=0, success=False,
                completed_at=self.get_sim_timestamp(), blocking_fact_ids=[blocked.id],
            ))
            plan = agent.mental_state.life_plan
            if plan and decision.get("plan_step_id") == (plan.current_step.id if plan.current_step else ""):
                await self._block_plan(agent, plan, blocked.id, "; ".join(feasibility.reasons))
            await self._finish_interaction(
                str(decision.get("interaction_id", "")), "blocked", fact_id=blocked.id,
            )
            return [self._make_event({
                "type": "action_blocked", "factId": blocked.id,
                "interactionId": decision.get("interaction_id", ""),
                "agentIds": [agent.id], "location": agent.state.current_location,
                "content": f"{agent.name}的行动未能执行：{'；'.join(feasibility.reasons)}",
            })]
        decision = feasibility.resolved_action
        task_id = str(decision.get("task_id", ""))
        if task_id:
            self.tasks.start(task_id, self.get_sim_timestamp())
        decision["started_at"] = self.get_sim_timestamp()
        decision["validated_effects"] = feasibility.expected_effects
        decision["duration_minutes"] = feasibility.estimated_minutes
        await self._start_interaction(agent, decision, feasibility.expected_effects)
        agent.mental_state.register_expectation(ActionExpectation(
            action_id=decision["action_id"], option_id=str(decision.get("option_id", "selected")),
            expected_effects=list(decision.get("predicted_effects") or feasibility.expected_effects),
            expected_duration=feasibility.estimated_minutes,
            assumptions=[str(item) for item in decision.get("assumptions", [])],
            created_at=self.get_sim_timestamp(),
            goal_ids=[str(item) for item in decision.get("supports_goal_ids", [])],
        ))
        intention = agent.mental_state.relevant_intention(decision)
        if intention:
            agent.mental_state.transition_intention(
                intention.id, "executing", self.get_sim_timestamp()
            )
        try:
            result = await agent.execute_action(decision, self)
        except Exception as exc:
            await self.trace.log(
                sim_time_str, agent.id, "action", "execute_error", str(exc)
            )
            result = None
        if not result:
            if task_id:
                self.tasks.block(task_id, "执行器未能启动已验证行动")
            blocked = self._record_fact(
                agent, "action_blocked", agent.state.current_location,
                {"action": decision, "reasons": ["执行器未能启动已验证行动"]},
                interaction_id=str(decision.get("interaction_id", "")),
            )
            plan = agent.mental_state.life_plan
            if plan and decision.get("plan_step_id") == (plan.current_step.id if plan.current_step else ""):
                await self._block_plan(agent, plan, blocked.id, "执行器未能启动已验证行动")
            agent.mental_state.record_outcome(ActionOutcome(
                action_id=str(decision.get("action_id", "")), actual_effects=[],
                duration=0, success=False, completed_at=self.get_sim_timestamp(),
                blocking_fact_ids=[blocked.id],
            ))
            await self._finish_interaction(
                str(decision.get("interaction_id", "")), "blocked", fact_id=blocked.id,
            )
            return [self._make_event({
                "type": "action_blocked", "factId": blocked.id,
                "interactionId": decision.get("interaction_id", ""),
                "agentIds": [agent.id], "location": agent.state.current_location,
                "content": f"{agent.name}的行动未能启动",
            })]
        plan = agent.mental_state.life_plan
        if plan and decision.get("plan_step_id") == (plan.current_step.id if plan.current_step else ""):
            plan.start_current_step(self.get_sim_timestamp())
        if result.get("type") == "dialogue_intent":
            self._dialogue_intents.append({
                "initiator": agent,
                "target_id": result.get("target"),
                "location": result.get("location"),
                "content": result.get("content", ""),
            })
            await self.trace.log(
                sim_time_str, agent.id, "dialogue", "intent_queued",
                result.get("content", "")[:200],
            )
            return []
        result["interactionId"] = decision.get("interaction_id", "")
        fact = self._record_fact(
            agent, result.get("type", "action"),
            result.get("location", agent.state.current_location),
            {
                "content": result.get("content", ""),
                "cause": result.get("cause", ""),
                "outcome": result.get("outcome", ""),
                "goal": agent.mental_state.active_goal,
            },
            participants=[item for item in result.get("agentIds", []) if item != agent.id],
            interaction_id=decision.get("interaction_id", ""),
        )
        result["factId"] = fact.id
        event = self._make_event(result)
        await self.trace.log(sim_time_str, agent.id, "event", event["type"], event["content"][:200])
        return [event]

    async def _handle_arrival_perception(self, agent: Agent,
                                         sim_time_str: str) -> list[dict]:
        """Trigger perception when an agent arrives at a new location.

        Runs perception → association → interrupt check chain.
        Returns an interrupt event if associations warrant it,
        otherwise a standard perception event.
        """
        try:
            stimulus = agent.perceive(self, as_stimulus=True)
            await self.trace.log(sim_time_str, agent.id, "perceive",
                                  "perception",
                                  str(stimulus)[:300])

            # ── Associate & interrupt check ──
            associations = await agent.associate(
                stimulus, self.embedding_provider, self.memory,
                current_sim_timestamp=self.get_sim_timestamp(),
            )
            current_plan = agent.state.current_action
            if agent.should_interrupt(associations, current_plan):
                await self.trace.log(
                    sim_time_str, agent.id, "perceive",
                    "association_interrupt",
                    f"Arrival interrupt: {stimulus.get('description')}",
                )
                event = {
                    "id": f"evt_{uuid.uuid4().hex[:8]}",
                    "time": f"{self.hour:02d}:{self.minute:02d}",
                    "type": "association_interrupt",
                    "agentIds": [agent.id],
                    "location": agent.state.current_location,
                    "content": f"{agent.name}到达{location_name(agent.state.current_location)}后触发了记忆联想",
                }
                return [event]
            event = {
                "id": f"evt_{uuid.uuid4().hex[:8]}",
                "time": f"{self.hour:02d}:{self.minute:02d}",
                "type": "perception",
                "agentIds": [agent.id],
                "location": agent.state.current_location,
                "content": stimulus.get("summary",
                                          f"{agent.name}观察了周围环境"),
            }
            return [event]
        except Exception as e:
            await self.trace.log(sim_time_str, agent.id, "perceive",
                                  "perception_error", str(e))
            return []

    def _check_moving_encounter(self, agent: Agent) -> dict | None:
        """Check if this MOVING agent is at the same cell as another MOVING agent.

        Returns an encounter event dict if found, None otherwise.
        Duplicate pairs (already reported) are skipped via _encountered_pairs.
        """
        ax, ay = agent.state.x, agent.state.y
        for other in self.agents:
            if other.id == agent.id:
                continue
            if other.state.status != "MOVING":
                continue
            if other.state.x == ax and other.state.y == ay:
                pair = tuple(sorted([agent.id, other.id]))
                if pair in self._encountered_pairs:
                    continue
                self._encountered_pairs.add(pair)
                return {
                    "id": f"evt_{uuid.uuid4().hex[:8]}",
                    "time": f"{self.hour:02d}:{self.minute:02d}",
                    "type": "encounter",
                    "agentIds": [agent.id, other.id],
                    "location": agent.state.current_location,
                    "content": f"{agent.name}和{other.name}在路上相遇了",
                }
        return None

    async def _detect_encounters(self, sim_time_str: str) -> list[dict]:
        """Detect agents at the same cell, interrupt ACTING ones on encounter."""
        events: list[dict] = []
        by_cell: dict[tuple[int, int], list[Agent]] = {}
        for agent in self.agents:
            pos = (agent.state.x, agent.state.y)
            if pos not in by_cell:
                by_cell[pos] = []
            by_cell[pos].append(agent)

        current_pairs: set[tuple[str, str]] = set()
        for pos, agents_at in by_cell.items():
            if len(agents_at) < 2:
                continue
            for i in range(len(agents_at)):
                for j in range(i + 1, len(agents_at)):
                    a, b = agents_at[i], agents_at[j]
                    pair = tuple(sorted([a.id, b.id]))
                    current_pairs.add(pair)
                    if pair in self._encountered_pairs:
                        continue
                    self._encountered_pairs.add(pair)
                    events.append(self._make_event({
                        "type": "encounter", "agentIds": [a.id, b.id],
                        "location": a.state.current_location,
                        "content": f"{a.name}和{b.name}相遇",
                    }))
                    self._queue_encounter_dialogue(a, b)
        self._active_encounter_pairs = current_pairs
        return events

    async def _rank_dialogue_topics(self, initiator: Agent, target: Agent,
                                    intents: list[dict]) -> dict | None:
        """Rank openings by shared memories first, relationship context second."""
        if not intents:
            return None
        relationship_a, relationship_b = await self.relationship_store.get_bidirectional(
            initiator.id, target.id,
        )
        shared = await self.memory.get_shared_between(initiator.id, target.id, limit=12)
        ranked = []
        for intent in intents:
            intention_id = str(intent.get("memory_intention_id", ""))
            source_ids = set()
            if intention_id:
                intention = next(
                    (item for item in initiator.mental_state.intentions if item.id == intention_id),
                    None,
                )
                source_ids = set(intention.source_ids if intention else [])
            linked = [memory for memory in shared if memory.id in source_ids]
            if not linked and intention_id:
                linked = shared[:1]
            memory_score = sum(
                0.18 + memory.importance * 0.07
                + (0.22 if memory.unresolved else 0.0)
                + memory.formation_score * 0.18
                for memory in linked
            )
            relationship_score = (
                relationship_a.familiarity * 0.035
                + max(0.0, relationship_a.affinity) * 0.025
                + relationship_a.trust * 0.02
                + relationship_b.familiarity * 0.015
            )
            score = memory_score + relationship_score + (0.25 if intention_id else 0.0)
            ranked.append((score, intent, linked))
        score, selected, linked = max(ranked, key=lambda item: item[0])
        selected["topic"] = {
            "relevanceScore": round(score, 3),
            "sharedMemoryIds": [memory.id for memory in linked],
            "relationship": {
                "familiarity": round((relationship_a.familiarity + relationship_b.familiarity) / 2, 2),
                "affinity": round((relationship_a.affinity + relationship_b.affinity) / 2, 2),
                "trust": round((relationship_a.trust + relationship_b.trust) / 2, 2),
            },
        }
        return selected

    def _queue_encounter_dialogue(self, initiator: Agent, target: Agent) -> None:
        """Turn a new idle encounter into one bounded, location-aware greeting."""
        location = initiator.state.current_location
        if (
            location == "in_transit"
            or target.state.current_location != location
            or initiator.state.status != "IDLE"
            or target.state.status != "IDLE"
            or initiator.id in self.dialogue.participant_ids()
            or target.id in self.dialogue.participant_ids()
        ):
            return
        if any(
            {intent.get("initiator").id, intent.get("target_id")} == {initiator.id, target.id}
            for intent in self._dialogue_intents
            if intent.get("initiator")
        ):
            return
        relevant = next((item for item in initiator.mental_state.open_intentions(self.get_sim_timestamp())
                         if item.source == "memory" and item.target_location in {"", location}), None)
        content = relevant.description if relevant else f"在{location_name(location)}碰见了，和对方打个招呼。"
        self._dialogue_intents.append({
            "initiator": initiator,
            "target_id": target.id,
            "location": location,
            "content": content,
            "memory_intention_id": relevant.id if relevant else "",
        })

    def _make_event(self, result: dict) -> dict:
        """Wrap an agent action result into a broadcast event."""
        return {
            "id": f"evt_{uuid.uuid4().hex[:8]}",
            "time": f"{self.hour:02d}:{self.minute:02d}",
            "simTime": self.get_sim_time_str(),
            "simTimestamp": self.get_sim_timestamp(),
            **result,
        }

    def _find_agent_by_id(self, agent_id: str) -> Agent | None:
        """Look up an agent by id."""
        for a in self.agents:
            if a.id == agent_id:
                return a
        return None

    # ═══════════════════════════════════════════════════════════════
    # Group-chat dialogue orchestration
    # ═══════════════════════════════════════════════════════════════

    async def _finalize_dialogue_events(self, raw_events: list[dict], before: dict) -> list[dict]:
        """Record dialogue facts and restore agent states after a completed turn."""
        for raw_event in raw_events:
            if raw_event.get("type") == "dialogue_end":
                await self._finish_interaction(
                    str(raw_event.get("interactionId", raw_event.get("dialogueId", ""))),
                    "completed",
                )
                continue
            if raw_event.get("type") != "dialogue_line":
                continue
            speaker_name = raw_event.get("speaker", "")
            speaker = next((agent for agent in self.agents if agent.name == speaker_name), None)
            if not speaker:
                continue
            fact = self._record_fact(
                speaker, "dialogue_line", raw_event.get("location", speaker.state.current_location),
                {"content": raw_event.get("content", ""), "dialogue_id": raw_event.get("dialogueId", "")},
                participants=[item for item in raw_event.get("agentIds", []) if item != speaker.id],
                source_ids=[raw_event.get("dialogueId", "")],
                interaction_id=raw_event.get("interactionId", raw_event.get("dialogueId", "")),
            )
            raw_event["factId"] = fact.id
            await self._finish_interaction(
                str(raw_event.get("interactionId", raw_event.get("dialogueId", ""))),
                "executing", fact_id=fact.id,
            )
        events = [self._make_event(event) for event in raw_events]
        active_ids = {session.id for session in self.dialogue._active_dialogues.values()}
        for dialogue_id, participants in before.items():
            if dialogue_id in active_ids:
                continue
            for participant in participants:
                current = participant.get_current_commitment(self.day, self.hour, self.minute)
                current_id = current.get("id") if current else None
                if not participant.resume_interrupted_activity(current_id):
                    participant.state.status = "IDLE"
        for session in self.dialogue._active_dialogues.values():
            for index, participant in enumerate(session.participants):
                participant.state.status = "SPEAKING" if index == session.current_speaker_index else "LISTENING"
        return events

    async def _advance_dialogue_turn_in_background(self, session, sim_time_str: str):
        before = {session.id: list(session.original_participants)}
        raw_events = await self.dialogue.advance_one_turn(session, self.trace, sim_time_str)
        return raw_events, before

    async def _advance_dialogues_realtime(self, sim_time_str: str) -> list[dict]:
        """Schedule one turn per dialogue and collect only responses already ready."""
        events: list[dict] = []
        ready_locations = [
            location for location, (_, task) in self._dialogue_turn_tasks.items() if task.done()
        ]
        for location in ready_locations:
            generation, task = self._dialogue_turn_tasks.pop(location)
            request = self._dialogue_request_states.pop(location, None)
            if generation != self._task_generation:
                continue
            if request:
                reason = dialogue_stale_reason(self, request)
                if reason:
                    await self.trace.log(
                        sim_time_str, "system", "dialogue", "turn_stale", reason,
                    )
                    continue
            try:
                raw_events, before = task.result()
            except asyncio.CancelledError:
                continue
            except Exception as exc:
                await self.trace.log(sim_time_str, "system", "dialogue", "turn_failed", str(exc)[:200])
                continue
            events.extend(await self._finalize_dialogue_events(raw_events, before))

        for location, session in list(self.dialogue._active_dialogues.items()):
            if location in self._dialogue_turn_tasks:
                continue
            task = asyncio.create_task(
                self._advance_dialogue_turn_in_background(session, sim_time_str),
                name=f"town-dialogue-{session.id}",
            )
            self._dialogue_turn_tasks[location] = (self._task_generation, task)
            self._dialogue_request_states[location] = capture_dialogue_request(
                self, session,
            )
        return events

    async def _advance_dialogues(self, sim_time_str: str) -> list[dict]:
        before = {
            session.id: list(session.original_participants)
            for session in self.dialogue._active_dialogues.values()
        }
        raw_events = await self.dialogue.advance_all(self.trace, sim_time_str)
        return await self._finalize_dialogue_events(raw_events, before)

    async def _run_group_dialogues(self, sim_time_str: str) -> list[dict]:
        """Start new dialogue sessions from intents without completing them."""
        events: list[dict] = []
        if not self._dialogue_intents:
            return events

        intents_by_location: dict[str, list[dict]] = {}
        for intent in self._dialogue_intents:
            intents_by_location.setdefault(intent["location"], []).append(intent)

        active_participants = self.dialogue.participant_ids()
        for loc, intents in intents_by_location.items():
            if self.dialogue.get_active_at_location(loc):
                continue
            participants: list[Agent] = []
            seen_ids: set[str] = set()
            opening_line = ""
            selected_topic = None
            for intent in intents:
                initiator = intent["initiator"]
                target = self._find_agent_by_id(intent["target_id"])
                for participant in (initiator, target):
                    if (participant and participant.id not in seen_ids
                            and participant.id not in active_participants
                            and participant.state.current_location == loc):
                        participants.append(participant)
                        seen_ids.add(participant.id)
                if not opening_line and intent["content"]:
                    opening_line = intent["content"]
            if len(participants) < 2:
                continue
            selected_topic = await self._rank_dialogue_topics(
                participants[0], participants[1], intents,
            )
            if selected_topic:
                opening_line = selected_topic.get("content", "")
            if not opening_line:
                continue

            for participant in participants:
                if participant.state.status == "ACTING" and participant._activity_ticks > 0:
                    desc = participant.interrupt_activity()
                    await self.trace.log(
                        sim_time_str, participant.id, "state", "activity_interrupted",
                        f"Interrupted during: {desc}",
                    )
            try:
                for index, participant in enumerate(participants):
                    participant.state.status = "SPEAKING" if index == 0 else "LISTENING"
                raw_events = await self.dialogue.start_dialogue(
                    participants, opening_line, loc, self.trace, sim_time_str,
                )
                dialogue_start = next(
                    (event for event in raw_events if event.get("type") == "dialogue_start"),
                    None,
                )
                if dialogue_start:
                    dialogue_id = str(dialogue_start.get("dialogueId", ""))
                    self._interactions[dialogue_id] = InteractionRecord(
                        id=dialogue_id,
                        sim_timestamp=self.get_sim_timestamp(),
                        state_version=self.state_version,
                        initiator_id=participants[0].id,
                        target_type="agent",
                        target_id=participants[1].id,
                        interaction_type="communicate",
                        status="executing",
                        payload={"opening_line": opening_line},
                    )
                    await self.interaction_store.upsert(self._interactions[dialogue_id])
                trigger_intent = next((intent.get("memory_intention_id", "") for intent in intents if intent.get("memory_intention_id")), "")
                if trigger_intent or (selected_topic and selected_topic.get("topic")):
                    for event in raw_events:
                        event["trigger"] = {
                            "type": "memory_intention" if trigger_intent else "ranked_topic",
                            **({"intentionId": trigger_intent} if trigger_intent else {}),
                            **(selected_topic.get("topic", {}) if selected_topic else {}),
                        }
                line_event = next((event for event in raw_events if event.get("type") == "dialogue_line"), None)
                if line_event:
                    speaker = participants[0]
                    fact = self._record_fact(
                        speaker, "dialogue_line", loc,
                        {"content": line_event.get("content", ""), "dialogue_id": line_event.get("dialogueId", "")},
                        participants=[participant.id for participant in participants if participant.id != speaker.id],
                        source_ids=[line_event.get("dialogueId", "")],
                        interaction_id=line_event.get("interactionId", line_event.get("dialogueId", "")),
                    )
                    line_event["factId"] = fact.id
                    await self._finish_interaction(
                        str(line_event.get("interactionId", line_event.get("dialogueId", ""))),
                        "executing", fact_id=fact.id,
                    )
                events.extend(self._make_event(event) for event in raw_events)
            except Exception as exc:
                for participant in participants:
                    participant.state.status = "IDLE"
                await self.trace.log(
                    sim_time_str, "system", "dialogue", "start_error", str(exc)[:300],
                )
        return events

    # ═══════════════════════════════════════════════════════════════
    # Periodic reflection
    # ═══════════════════════════════════════════════════════════════

    async def _run_reflections(self, sim_time_str: str):
        """Create higher-level beliefs from a meaningful simulated-time window."""
        end_timestamp = self.get_sim_timestamp()
        start_timestamp = self._last_reflection_timestamp + 1
        for agent in self.agents:
            try:
                recent = await self.memory.get_in_sim_range(
                    agent.id, start_timestamp, end_timestamp,
                    exclude_reflections=True,
                )
                recent = [
                    memory for memory in recent
                    if memory.type in ("dialogue", "observation") and memory.importance >= 6
                ]
                if len(recent) < 2 or sum(memory.importance for memory in recent) < 14:
                    continue
                memory_text = "\n".join(
                    f"- [{memory.time}] {memory.content}" for memory in recent
                )
                reflections = await self.llm.reflect(agent.persona_text(), memory_text)
                source_ids = [memory.id for memory in recent]
                for reflection in reflections[:1]:
                    await self.memory.add(
                        agent.id, agent.state.current_location,
                        reflection, "reflection",
                        auto_importance(reflection, "reflection"),
                        sim_time=sim_time_str,
                        source_ids=source_ids,
                    )
                    await self.trace.log(
                        sim_time_str, agent.id, "state", "reflection",
                        reflection[:200],
                    )
            except Exception as exc:
                await self.trace.log(
                    sim_time_str, agent.id, "error", "reflection_failed", str(exc)[:200]
                )

    async def _daily_maintenance(self):
        """Run per-day memory evolution: decay and merge for all agents.

        Called once per simulated day when the clock wraps from
        SIM_END_HOUR to SIM_START_HOUR (overnight: 23→6 transition).
        """
        for agent in self.agents:
            agent.reset_commitments_for_new_day()
            self.habits.decay(agent.id, self.get_sim_timestamp())
            try:
                await self.memory.apply_decay(agent.id, self.get_sim_timestamp())
            except Exception as exc:
                await self.trace.log(
                    self.get_sim_time_str(), agent.id, "error",
                    "memory_decay_failed", str(exc)[:200],
                )
            try:
                await self.memory.merge_similar(agent.id)
            except Exception as exc:
                await self.trace.log(
                    self.get_sim_time_str(), agent.id, "error",
                    "memory_merge_failed", str(exc)[:200],
                )
