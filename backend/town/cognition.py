"""Auditable runtime cognition shared by action, dialogue, and memory."""

from dataclasses import asdict, dataclass, field
import uuid

GOAL_STATES = {"proposed", "active", "blocked", "suspended", "completed", "failed", "abandoned"}
INTENTION_STATES = {"open", "executing", "blocked", "suspended", "completed", "abandoned", "expired"}
BELIEF_STATES = {"observed", "reported", "inferred", "uncertain", "disproven"}
INTENTION_INTERACTION_TYPES = {
    "move", "consume", "rest", "communicate", "inspect", "request_service",
    "take", "put", "transfer", "use_resource", "operate", "produce",
    "work_on_goal", "wait",
}


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass(frozen=True)
class ActiveConsideration:
    """A code-owned reason that can influence the next decision."""

    kind: str
    source_id: str
    description: str
    priority: float
    urgency: float
    hard: bool = False
    target_location: str = ""
    affected_agents: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class IntentionProposal:
    """Compact intent proposal resolved into locations and actions later."""

    purpose: str
    interaction_type: str = "wait"
    target_agent_id: str = ""
    resource_kind: str = ""
    supports_goal_ids: list[str] = field(default_factory=list)
    urgency: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict | None) -> "IntentionProposal":
        raw = raw if isinstance(raw, dict) else {}
        purpose = str(raw.get("purpose", raw.get("description", ""))).strip()
        interaction_type = str(raw.get("interaction_type", "wait")).strip()
        if interaction_type not in INTENTION_INTERACTION_TYPES:
            interaction_type = "wait"
        goals = raw.get("supports_goal_ids", raw.get("goal_ids", []))
        if not isinstance(goals, list):
            goals = [goals] if goals else []
        try:
            urgency = float(raw.get("urgency", 0.0))
        except (TypeError, ValueError):
            urgency = 0.0
        return cls(
            purpose=purpose[:180],
            interaction_type=interaction_type,
            target_agent_id=str(raw.get("target_agent_id", raw.get("target", "")))[:80],
            resource_kind=str(raw.get("resource_kind", ""))[:80],
            supports_goal_ids=[str(item) for item in goals[:10]],
            urgency=max(0.0, min(1.0, urgency)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FactEvent:
    sim_time: str
    sim_timestamp: int
    type: str
    agent_id: str
    location: str
    details: dict
    participants: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    known_by: list[str] = field(default_factory=list)
    interaction_id: str = ""
    id: str = field(default_factory=lambda: _id("fact"))


@dataclass
class Goal:
    description: str
    source: str
    created_at: int
    importance: float = 0.5
    urgency: float = 0.3
    responsibility: float = 0.3
    flexibility: float = 0.7
    deadline: int | None = None
    affected_agents: list[str] = field(default_factory=list)
    success_conditions: list[dict] = field(default_factory=list)
    failure_conditions: list[dict] = field(default_factory=list)
    source_fact_ids: list[str] = field(default_factory=list)
    progress: float = 0.0
    status: str = "proposed"
    id: str = field(default_factory=lambda: _id("goal"))


@dataclass
class Belief:
    proposition: str
    confidence: float
    status: str
    learned_at: int
    last_confirmed_at: int
    source_fact_ids: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: _id("belief"))


@dataclass
class Intention:
    description: str
    source: str
    created_at: int
    target_location: str = ""
    interaction_type: str = "act"
    goal_id: str = ""
    expected_until: int | None = None
    confidence: float = 0.7
    status: str = "open"
    public: bool = False
    source_ids: list[str] = field(default_factory=list)
    progress: float = 0.0
    last_attempt_at: int | None = None
    blocking_fact_ids: list[str] = field(default_factory=list)
    completion_fact_ids: list[str] = field(default_factory=list)
    abandonment_reason: str = ""
    superseded_by: str = ""
    id: str = field(default_factory=lambda: _id("intent"))


@dataclass
class EmotionalState:
    label: str = "平静"
    valence: float = 0.0
    arousal: float = 0.1
    cause_fact_ids: list[str] = field(default_factory=list)


@dataclass
class EpisodeRecord:
    goal_id: str
    goal: str
    started_at: int
    started_sim_time: str
    fact_ids: list[str] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    last_updated_at: int = 0
    status: str = "active"
    id: str = field(default_factory=lambda: _id("episode"))


@dataclass
class PlanStep:
    description: str
    action: dict
    expected_outcome: list[dict] = field(default_factory=list)
    status: str = "pending"
    started_at: int | None = None
    completed_at: int | None = None
    blocking_fact_ids: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: _id("step"))


@dataclass
class LifePlan:
    focus: str
    motive: str
    created_at: int
    review_at: int
    goal_ids: list[str] = field(default_factory=list)
    steps: list[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    status: str = "active"
    replan_reason: str = ""
    id: str = field(default_factory=lambda: _id("plan"))
    purpose: str = ""
    experience_id: str = ""
    start_location: str = ""

    @property
    def current_step(self) -> PlanStep | None:
        while self.current_step_index < len(self.steps):
            step = self.steps[self.current_step_index]
            if step.status in {"completed", "skipped"}:
                self.current_step_index += 1
                continue
            return step
        return None

    def start_current_step(self, now: int) -> PlanStep:
        step = self.current_step
        if not step:
            raise RuntimeError("life plan has no pending step")
        if step.status == "executing":
            raise RuntimeError("life plan step is already executing")
        if step.status == "blocked":
            raise RuntimeError("life plan step is blocked")
        step.status = "executing"
        step.started_at = now
        return step

    def complete_step(self, step_id: str, now: int) -> bool:
        step = self.current_step
        if not step or step.id != step_id or step.status != "executing":
            return False
        step.status = "completed"
        step.completed_at = now
        self.current_step_index += 1
        if self.current_step is None:
            self.status = "completed"
        return True

    def block_current_step(self, fact_id: str, reason: str) -> None:
        step = self.current_step
        if step:
            step.status = "blocked"
            if fact_id:
                step.blocking_fact_ids.append(fact_id)
        self.status = "blocked"
        self.replan_reason = reason[:240]


@dataclass
class InteractionContext:
    agent_id: str
    dialogue_id: str
    ended_at: str
    ended_timestamp: int
    summary: str
    last_messages: list[str] = field(default_factory=list)


@dataclass
class ActionExpectation:
    action_id: str
    option_id: str
    expected_effects: list[dict]
    expected_duration: int
    assumptions: list[str]
    created_at: int
    goal_ids: list[str] = field(default_factory=list)


@dataclass
class ActionOutcome:
    action_id: str
    actual_effects: list[dict]
    duration: int
    success: bool
    completed_at: int
    blocking_fact_ids: list[str] = field(default_factory=list)
    prediction_error: float = 0.0


@dataclass
class SituationAssessment:
    confirmed_fact_ids: list[str] = field(default_factory=list)
    belief_ids: list[str] = field(default_factory=list)
    uncertainties: list[dict] = field(default_factory=list)
    active_motives: list[dict] = field(default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    relevant_goal_ids: list[str] = field(default_factory=list)
    relevant_intention_ids: list[str] = field(default_factory=list)


@dataclass
class ActionOption:
    id: str
    action: dict
    expected_effects: list[dict] = field(default_factory=list)
    risks: list[dict] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    supports_goal_ids: list[str] = field(default_factory=list)
    conflicts_with_goal_ids: list[str] = field(default_factory=list)


@dataclass
class DecisionProposal:
    assessment: SituationAssessment
    options: list[ActionOption]
    preferred_option_id: str
    mental_update: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "DecisionProposal":
        raw = raw if isinstance(raw, dict) else {}

        def items(value) -> list:
            if value is None:
                return []
            return value if isinstance(value, list) else [value]

        assessment_raw = raw.get("assessment") or raw.get("situation_assessment") or {}
        if not isinstance(assessment_raw, dict):
            assessment_raw = {"uncertainties": [{"question": str(assessment_raw)}]}
        assessment = SituationAssessment(
            confirmed_fact_ids=[str(x) for x in items(assessment_raw.get("confirmed_fact_ids", assessment_raw.get("confirmed_facts", [])))],
            belief_ids=[str(x) for x in items(assessment_raw.get("belief_ids", []))],
            uncertainties=[x if isinstance(x, dict) else {"question": str(x)} for x in items(assessment_raw.get("uncertainties", []))],
            active_motives=[x if isinstance(x, dict) else {"description": str(x)} for x in items(assessment_raw.get("active_motives", []))],
            constraints=[x if isinstance(x, dict) else {"description": str(x)} for x in items(assessment_raw.get("constraints", []))],
            relevant_goal_ids=[str(x) for x in items(assessment_raw.get("relevant_goal_ids", []))],
            relevant_intention_ids=[str(x) for x in items(assessment_raw.get("relevant_intention_ids", []))],
        )
        options = []
        for index, option in enumerate(raw.get("options", [])):
            if not isinstance(option, dict) or not isinstance(option.get("action"), dict):
                continue
            options.append(ActionOption(
                id=str(option.get("id") or f"option_{index + 1}"),
                action=dict(option["action"]),
                expected_effects=[x if isinstance(x, dict) else {"description": str(x)} for x in items(option.get("expected_effects", option.get("expected_outcome", [])))],
                risks=[x if isinstance(x, dict) else {"description": str(x)} for x in items(option.get("risks", []))],
                assumptions=[str(x) for x in items(option.get("assumptions", []))],
                supports_goal_ids=[str(x) for x in option.get("supports_goal_ids", option.get("supports", []))],
                conflicts_with_goal_ids=[str(x) for x in option.get("conflicts_with_goal_ids", [])],
            ))
        return cls(
            assessment=assessment,
            options=options,
            preferred_option_id=str(raw.get("preferred_option_id", "")),
            mental_update=raw.get("mental_update") if isinstance(raw.get("mental_update"), dict) else {},
        )


@dataclass
class MentalState:
    goals: list[Goal] = field(default_factory=list)
    intentions: list[Intention] = field(default_factory=list)
    beliefs: list[Belief] = field(default_factory=list)
    attention: list[str] = field(default_factory=list)
    emotion: EmotionalState = field(default_factory=EmotionalState)
    current_episode: EpisodeRecord | None = None
    life_plan: LifePlan | None = None
    recent_interactions: dict[str, InteractionContext] = field(default_factory=dict)
    recent_fact_ids: list[str] = field(default_factory=list)
    expectations: dict[str, ActionExpectation] = field(default_factory=dict)
    outcomes: list[ActionOutcome] = field(default_factory=list)
    last_decision: dict = field(default_factory=dict)

    def complete_plan_step(self, step_id: str, now: int) -> dict | None:
        plan = self.life_plan
        if not plan or not plan.complete_step(step_id, now) or plan.status != "completed":
            return
        if (len(plan.steps) < 2 or any(step.status != "completed" for step in plan.steps)
                or not any(step.action.get("interaction_type") not in {"move", "wait", "inspect"}
                           for step in plan.steps)):
            return
        # The store owns action sanitization and structural deduplication.
        actions = [dict(step.action) for step in plan.steps]
        return {
            "purpose": plan.purpose or plan.focus, "focus": plan.focus,
            "start_location": plan.start_location,
            "actions": actions, "completed_at": now,
        }

    @property
    def active_goal_object(self) -> Goal | None:
        active = [goal for goal in self.goals if goal.status == "active"]
        return max(active, key=lambda goal: goal.importance + goal.urgency + goal.responsibility, default=None)

    @property
    def active_goal(self) -> str:
        goal = self.active_goal_object
        return goal.description if goal else ""

    @property
    def goal_source(self) -> str:
        goal = self.active_goal_object
        return goal.source if goal else ""

    def add_goal(self, goal: Goal) -> Goal:
        goal.status = goal.status if goal.status in GOAL_STATES else "proposed"
        for existing in self.goals:
            if existing.description == goal.description and existing.status not in {"completed", "failed", "abandoned"}:
                existing.importance = max(existing.importance, goal.importance)
                existing.urgency = max(existing.urgency, goal.urgency)
                existing.deadline = goal.deadline or existing.deadline
                if goal.status == "active":
                    existing.status = "active"
                return existing
        if goal.status == "proposed" and self.active_goal_object is None:
            goal.status = "active"
        self.goals.append(goal)
        self.goals = self.goals[-20:]
        return goal

    def transition_goal(self, goal_id: str, status: str, progress: float | None = None):
        if status not in GOAL_STATES:
            return
        for goal in self.goals:
            if goal.id == goal_id:
                goal.status = status
                if progress is not None:
                    goal.progress = max(0.0, min(1.0, progress))
                return

    def add_belief(self, belief: Belief) -> Belief:
        belief.status = belief.status if belief.status in BELIEF_STATES else "uncertain"
        for existing in self.beliefs:
            if existing.proposition == belief.proposition and existing.status != "disproven":
                existing.confidence = belief.confidence
                existing.status = belief.status
                existing.last_confirmed_at = belief.last_confirmed_at
                existing.source_fact_ids = list(dict.fromkeys(existing.source_fact_ids + belief.source_fact_ids))
                return existing
        self.beliefs.append(belief)
        self.beliefs = self.beliefs[-50:]
        return belief

    def add_fact(self, fact: FactEvent, goal: str = ""):
        self.recent_fact_ids = (self.recent_fact_ids + [fact.id])[-40:]
        goal_obj = self.active_goal_object
        episode_goal = goal or (goal_obj.description if goal_obj else "") or fact.details.get("goal", "")
        if not episode_goal:
            episode_goal = fact.details.get("activity") or fact.details.get("purpose") or fact.type
        goal_id = goal_obj.id if goal_obj and goal_obj.description == episode_goal else ""
        if (self.current_episode and self.current_episode.goal != episode_goal
                and fact.type in {"activity_started", "movement_arrived"}
                and fact.sim_timestamp - self.current_episode.last_updated_at >= 5):
            self.current_episode.status = "completed"
            self.current_episode = None
        if self.current_episode is None:
            self.current_episode = EpisodeRecord(
                goal_id=goal_id, goal=episode_goal, started_at=fact.sim_timestamp,
                started_sim_time=fact.sim_time, last_updated_at=fact.sim_timestamp,
            )
        if fact.id not in self.current_episode.fact_ids:
            self.current_episode.fact_ids.append(fact.id)
        self.current_episode.participants = list(dict.fromkeys(self.current_episode.participants + fact.participants))
        self.current_episode.last_updated_at = fact.sim_timestamp

    def apply_update(self, update: dict | None, now: int, source_ids: list[str] | None = None,
                     valid_fact_ids: set[str] | None = None):
        if not update:
            return
        source_ids = list(source_ids or [])
        valid_fact_ids = set(valid_fact_ids) if valid_fact_ids is not None else None
        active_goal = str(update.get("active_goal", "")).strip()
        if active_goal:
            previous = self.active_goal
            goal = self.add_goal(Goal(
                description=active_goal[:160], source=str(update.get("goal_source", "llm"))[:40],
                created_at=now, importance=float(update.get("goal_importance", 0.6)),
                urgency=float(update.get("goal_urgency", 0.4)), source_fact_ids=source_ids,
                status="active",
            ))
            for other in self.goals:
                if other.id != goal.id and other.status == "active":
                    other.status = "suspended"
            if self.current_episode and previous and previous != active_goal:
                self.current_episode.status = "completed"
                self.current_episode = None
        if isinstance(update.get("attention"), list):
            self.attention = [str(item)[:80] for item in update["attention"][:5] if str(item).strip()]
        emotion = update.get("emotion")
        if isinstance(emotion, dict):
            self.emotion = EmotionalState(
                label=str(emotion.get("label", self.emotion.label))[:20],
                valence=max(-1.0, min(1.0, float(emotion.get("valence", self.emotion.valence)))),
                arousal=max(0.0, min(1.0, float(emotion.get("arousal", self.emotion.arousal)))),
                cause_fact_ids=[str(item) for item in emotion.get("cause_fact_ids", [])[:6]],
            )
        for raw in update.get("belief_updates", []) if isinstance(update.get("belief_updates"), list) else []:
            if isinstance(raw, str):
                raw = {"proposition": raw, "status": "inferred", "confidence": 0.5}
            if isinstance(raw, dict) and str(raw.get("proposition", "")).strip():
                raw_sources = raw.get("source_fact_ids", source_ids)
                raw_sources = raw_sources if isinstance(raw_sources, list) else [raw_sources]
                belief_sources = [str(x) for x in raw_sources]
                if valid_fact_ids is not None:
                    belief_sources = [fact_id for fact_id in belief_sources if fact_id in valid_fact_ids]
                status = str(raw.get("status", "inferred"))
                confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
                # A belief without an auditable fact remains useful as a
                # hypothesis, but must not enter the state as an observation.
                if not belief_sources:
                    status = "uncertain"
                    confidence = min(confidence, 0.35)
                self.add_belief(Belief(
                    proposition=str(raw["proposition"])[:240], confidence=confidence,
                    status=status, learned_at=now, last_confirmed_at=now,
                    source_fact_ids=belief_sources,
                ))
        for raw in update.get("intentions", []) if isinstance(update.get("intentions"), list) else []:
            if isinstance(raw, str):
                raw = {"description": raw, "source": "llm"}
            if not isinstance(raw, dict) or not str(raw.get("description", "")).strip():
                continue
            self.add_intention(Intention(
                description=str(raw["description"])[:180], source=str(raw.get("source", "llm"))[:40],
                created_at=now, target_location=str(raw.get("target_location", ""))[:40],
                interaction_type=str(raw.get("interaction_type", "act"))[:30], goal_id=str(raw.get("goal_id", "")),
                expected_until=int(raw["expected_until"]) if raw.get("expected_until") is not None else None,
                confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.7)))),
                public=bool(raw.get("public", False)), source_ids=source_ids,
            ))
        for transition in update.get("intention_transitions", []) if isinstance(update.get("intention_transitions"), list) else []:
            if isinstance(transition, dict):
                self.transition_intention(str(transition.get("id", "")), str(transition.get("status", "")), now,
                                          reason=str(transition.get("reason", "")))

    def add_intention(self, intention: Intention) -> Intention:
        intention.status = intention.status if intention.status in INTENTION_STATES else "open"
        for existing in self.intentions:
            if existing.status in {"open", "executing", "suspended"} and existing.description == intention.description:
                existing.confidence = max(existing.confidence, intention.confidence)
                existing.expected_until = intention.expected_until or existing.expected_until
                return existing
        self.intentions.append(intention)
        self.intentions = self.intentions[-20:]
        return intention

    def transition_intention(self, intention_id: str, status: str, now: int,
                             fact_id: str = "", reason: str = ""):
        if status not in INTENTION_STATES:
            return
        for intention in self.intentions:
            if intention.id != intention_id:
                continue
            intention.status = status
            intention.last_attempt_at = now
            if status == "executing":
                intention.progress = max(0.05, intention.progress)
            elif status == "completed":
                intention.progress = 1.0
                if fact_id:
                    intention.completion_fact_ids.append(fact_id)
            elif status == "blocked" and fact_id:
                intention.blocking_fact_ids.append(fact_id)
            elif status == "abandoned":
                intention.abandonment_reason = reason
            return

    def relevant_intention(self, action: dict) -> Intention | None:
        """Choose the strongest intention supported by an action."""
        action_type = str(action.get("interaction_type") or action.get("action") or "")
        location = str(action.get("location") or action.get("target_location") or "")
        goal_id = str(action.get("goal_id") or "")
        intention_id = str(action.get("intention_id") or "")
        candidates = []
        for index, intention in enumerate(self.intentions):
            if intention.status not in {"open", "suspended", "blocked"}:
                continue
            score = 0
            if intention.id == intention_id:
                score += 8
            if goal_id and intention.goal_id == goal_id:
                score += 5
            if action_type and intention.interaction_type == action_type:
                score += 3
            if location and intention.target_location == location:
                score += 3
            if score:
                candidates.append((score, index, intention))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))[2]

    def expire_intentions(self, now: int) -> list[str]:
        """Mark overdue intentions explicitly; reads do not mutate state."""
        expired = []
        for intention in self.intentions:
            if (
                intention.status in {"open", "suspended", "blocked"}
                and intention.expected_until is not None
                and now > intention.expected_until
            ):
                intention.status = "expired"
                expired.append(intention.id)
        return expired

    def open_intentions(self, now: int) -> list[Intention]:
        """Return intentions active at ``now`` without changing their state."""
        return [
            item for item in self.intentions
            if item.status in {"open", "executing", "blocked", "suspended"}
            and (item.expected_until is None or now <= item.expected_until)
        ]

    def register_expectation(self, expectation: ActionExpectation):
        self.expectations[expectation.action_id] = expectation

    def record_outcome(self, outcome: ActionOutcome):
        self.outcomes.append(outcome)
        self.outcomes = self.outcomes[-20:]
        expectation = self.expectations.pop(outcome.action_id, None)
        if expectation:
            expected_types = {str(item.get("type", item.get("description", ""))) for item in expectation.expected_effects}
            actual_types = {str(item.get("type", item.get("description", ""))) for item in outcome.actual_effects}
            union = expected_types | actual_types
            outcome.prediction_error = 0.0 if not union else 1.0 - len(expected_types & actual_types) / len(union)

    def set_recent_interaction(self, other_id: str, context: InteractionContext):
        self.recent_interactions[other_id] = context

    def summary_for(self, task_type: str, now: int) -> dict:
        """Return a bounded mental-state view for one LLM task.

        ``context_dict`` remains the complete diagnostic/API view. This method
        is intentionally read-only so a caller cannot change cognition merely
        by building a prompt.
        """
        task_type = str(task_type or "action_decision")
        limits = {
            "action_decision": (5, 5, 8, 3),
            "daily_plan": (8, 8, 10, 5),
            "life_plan": (6, 6, 8, 4),
            "routine_selection": (3, 3, 4, 2),
            "dialogue": (2, 4, 6, 3),
            "reflection": (6, 6, 10, 5),
            "memory_formation": (2, 3, 4, 2),
        }
        goal_limit, intention_limit, belief_limit, outcome_limit = limits.get(
            task_type, limits["action_decision"]
        )
        goals = [
            goal for goal in self.goals
            if goal.status in {"proposed", "active", "blocked", "suspended"}
        ]
        goals.sort(
            key=lambda goal: (
                goal.status == "active",
                goal.status == "blocked",
                goal.importance + goal.urgency + goal.responsibility,
            ),
            reverse=True,
        )
        intentions = self.open_intentions(now)
        intentions.sort(
            key=lambda intention: (
                intention.status == "executing",
                intention.status == "blocked",
                intention.confidence,
            ),
            reverse=True,
        )
        plan = None
        if self.life_plan:
            step = self.life_plan.current_step
            plan = {
                "id": self.life_plan.id,
                "focus": self.life_plan.focus,
                "motive": self.life_plan.motive,
                "status": self.life_plan.status,
                "review_at": self.life_plan.review_at,
                "experience_id": self.life_plan.experience_id,
                "current_step": asdict(step) if step else None,
            }
        result = {
            "active_goal": self.active_goal,
            "active_goal_detail": asdict(self.active_goal_object) if self.active_goal_object else None,
            "goals": [asdict(item) for item in goals[:goal_limit]],
            "intentions": [asdict(item) for item in intentions[:intention_limit]],
            "emotion": asdict(self.emotion),
            "attention": self.attention[:5],
            "life_plan": plan,
        }
        if task_type in {"action_decision", "dialogue", "reflection"}:
            result["recent_outcomes"] = [asdict(item) for item in self.outcomes[-outcome_limit:]]
            result["beliefs"] = [
                asdict(item) for item in self.beliefs[-belief_limit:]
                if item.status != "disproven"
            ]
        if task_type in {"action_decision", "dialogue"}:
            result["recent_interactions"] = {
                key: asdict(value) for key, value in list(self.recent_interactions.items())[-4:]
            }
            result["last_decision"] = self.last_decision
        return result

    def context_dict(self, now: int) -> dict:
        return {
            "active_goal": self.active_goal,
            "active_goal_detail": asdict(self.active_goal_object) if self.active_goal_object else None,
            "goals": [asdict(item) for item in self.goals if item.status in {"proposed", "active", "blocked", "suspended"}],
            "attention": self.attention,
            "emotion": asdict(self.emotion),
            "intentions": [asdict(item) for item in self.open_intentions(now)],
            "beliefs": [asdict(item) for item in self.beliefs[-20:] if item.status != "disproven"],
            "current_episode": asdict(self.current_episode) if self.current_episode else None,
            "life_plan": asdict(self.life_plan) if self.life_plan else None,
            "recent_interactions": {key: asdict(value) for key, value in self.recent_interactions.items()},
            "recent_outcomes": [asdict(item) for item in self.outcomes[-5:]],
            "last_decision": self.last_decision,
        }


class FactLedger:
    def __init__(self, max_facts: int = 2000):
        self.max_facts = max_facts
        self._facts: dict[str, FactEvent] = {}
        self._order: list[str] = []

    def add(self, fact: FactEvent) -> FactEvent:
        if not fact.known_by:
            fact.known_by = list(dict.fromkeys([fact.agent_id, *fact.participants]))
        self._facts[fact.id] = fact
        self._order.append(fact.id)
        while len(self._order) > self.max_facts:
            self._facts.pop(self._order.pop(0), None)
        return fact

    def get(self, fact_id: str) -> FactEvent | None:
        return self._facts.get(fact_id)

    def is_known_by(self, fact_id: str, agent_id: str, now: int | None = None) -> bool:
        fact = self.get(fact_id)
        return bool(fact and agent_id in fact.known_by and (now is None or fact.sim_timestamp <= now))

    def validate_references(self, fact_ids: list[str], agent_id: str, now: int) -> tuple[list[str], list[str]]:
        valid, invalid = [], []
        for fact_id in dict.fromkeys(fact_ids):
            (valid if self.is_known_by(fact_id, agent_id, now) else invalid).append(fact_id)
        return valid, invalid

    def many(self, fact_ids: list[str]) -> list[FactEvent]:
        return [self._facts[fact_id] for fact_id in fact_ids if fact_id in self._facts]

    def known_for(self, agent_id: str, since: int = 0, limit: int = 30) -> list[FactEvent]:
        facts = [self._facts[fid] for fid in reversed(self._order)
                 if agent_id in self._facts[fid].known_by and self._facts[fid].sim_timestamp >= since]
        return list(reversed(facts[:limit]))

    def recent_for(self, agent_id: str, since: int = 0, limit: int = 20) -> list[FactEvent]:
        return self.known_for(agent_id, since, limit)

    def clear(self):
        self._facts.clear()
        self._order.clear()
