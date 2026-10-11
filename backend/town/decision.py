"""Two-stage cognitive planning with fact and feasibility validation."""

from dataclasses import asdict
import json

from .cognition import DecisionProposal, IntentionProposal, LifePlan, PlanStep
from .daily_plan import DailyPlan, parse_daily_plan


class CognitivePlanner:
    def __init__(self, llm):
        self.llm = llm

    async def select_routine_candidate(self, agent, engine, candidates) -> dict:
        """Let the chat model choose only among code-generated candidates."""
        if not candidates:
            raise ValueError("routine selection requires candidates")
        candidate_map = {candidate.id: candidate for candidate in candidates}
        context = await engine.context.build(agent, "routine_selection", engine)
        raw = await self._json_call(
            """你是当前世界中的角色。系统已经生成并验证了可选的自由时间活动。
只能从给定candidate_id中选一个，不得发明地点、人物、行动或事实。
只输出JSON：{"candidate_id":"候选ID","brief_reason":"基于给定理由的简短选择依据","mental_update":{}}。
选择依据可以考虑当前需要、既有习惯、地点中的人物和今日生活方向，但不要复述内部数值。""",
            {**context, "candidates": [candidate.to_dict() for candidate in candidates]},
            mode="chat", task="routine_selection",
        )
        selected_id = str(raw.get("candidate_id", ""))
        selected = candidate_map.get(selected_id)
        if selected is None:
            raise ValueError("routine selection returned an unknown candidate")
        action = dict(selected.action)
        action["selection_reason"] = str(raw.get("brief_reason", ""))[:240]
        action["mental_update"] = raw.get("mental_update") if isinstance(raw.get("mental_update"), dict) else {}
        return action

    async def propose_intention(self, agent, engine, context: dict) -> dict:
        """Propose a purpose before the world query resolves locations."""
        raw = await self._json_call(
            """你负责为角色提出下一步意图，不直接执行行动。只输出JSON：
{"purpose":"角色想完成的事情","interaction_type":"wait","target_agent_id":"","resource_kind":"","supports_goal_ids":[],"urgency":0.0}
意图只描述目的和交互方向，不要填写地点、资源ID、设施ID或过程ID。
只能使用上下文中已有的目标、需求、事实和约束，不补写世界事实。
interaction_type只能是move, consume, rest, communicate, inspect, request_service, take, put, transfer, use_resource, operate, produce, work_on_goal, wait。""",
            context, mode="chat", task="intention_proposal",
        )
        proposal = IntentionProposal.from_dict(raw)
        if not proposal.purpose:
            raise RuntimeError("intention proposal returned no purpose")
        return proposal.to_dict()

    async def _resolve_intention_context(self, agent, engine, context: dict,
                                         task_type: str) -> tuple[dict, dict]:
        """Resolve purpose before asking the world for locations and resources."""
        intention = await self.propose_intention(agent, engine, context)
        resolved = await engine.context.build(
            agent, task_type, engine, intention=intention,
            sim_time_str=context.get("sim_time", engine.get_sim_time_str()),
        )
        resolved["selected_intention"] = intention
        return resolved, intention

    async def create_daily_plan(self, agent, engine, context: dict,
                                replan_reason: str = "") -> DailyPlan:
        """Create a stable coarse plan; executable actions are chosen later."""
        if self.llm.fallback:
            raise RuntimeError("daily planner unavailable in fallback mode")
        if context.get("request", {}).get("task_type") != "daily_plan":
            context = await engine.context.build(agent, "daily_plan", engine)
        raw = await self._json_call(
            """你负责为角色生成今天的大致生活安排，不要生成逐tick动作。只输出JSON：
{"focus":"今天最在意的事情","review_after_minutes":180,"blocks":[{"id":"morning","window":[360,720],"intent":"上午的生活意图","candidates":[{"interaction_type":"act","content":"旁观者可见的活动","location":"地点ID"}],"priority":0.5,"flexibility":0.7}]}
blocks必须是1至6个不重叠的时间区块，window使用当天的分钟数（0到1440）。
固定工作、上学、值班或已确认约定必须保留；弹性个人时间可以提供多个候选。
不要把所有候选都当成必须执行的任务，不要补写上下文不存在的事实。
每个候选只能描述一个可观察行动，地点必须来自上下文中的真实地点。
只输出JSON，不要输出思维过程。""",
            {**context, "replan_reason": replan_reason},
            mode="chat", task="daily_plan",
        )
        return parse_daily_plan(raw, agent.id, engine.day, engine.get_sim_timestamp())

    async def create_life_plan(self, agent, engine, context: dict,
                               replan_reason: str = "") -> LifePlan:
        if self.llm.fallback:
            raise RuntimeError("life planner unavailable in fallback mode")
        context, intention = await self._resolve_intention_context(
            agent, engine, context, "life_plan",
        )
        experiences = await engine.procedure_store.discover(agent, engine, intention)
        raw = await self._json_call(
            """你负责为角色形成一段连续生活计划，不逐tick重新决策。只输出JSON：
{"focus":"接下来一段时间在意的事情","motive":"角色自己的简短动机","review_after_minutes":60,"goal_ids":[],"steps":[{"description":"旁观者可见的自然行动","action":{"interaction_type":"move","content":"自然行动描述","location":"地点ID","duration_minutes":15},"expected_outcome":[]}]}
steps必须有1至6项，并按真实执行顺序排列。普通生活不需要列多个候选。
若给定experiences中的摘要适合当前意图，可输出{"experience_id":"经验ID"}由系统加载步骤；无合适经验则生成新steps。经验是历史方法，不能把历史结果当成当前事实。
interaction_type只能是move, consume, rest, communicate, inspect, request_service, take, put, transfer, use_resource, operate, produce, work_on_goal, wait。
move必须填写known_locations中的location ID。work_on_goal必须填写当前有效goal_id与progress_delta。环境操作必须引用上下文中可见的结构化ID。request_service只能在当前位置、完成inspect后请求可见服务台提供的真实商品；成功后商品进入背包，随后才能consume。
计划要有连续性：准备、移动和到达后的活动应是不同步骤。不要用等待、观察、休息填充时间；只有角色确实在等待某个条件、需要观察未知信息或需要恢复时才能安排。
content不得包含ID、括号假设或系统校验说明。不要补写上下文没有的事实。""",
            {**context, "replan_reason": replan_reason, "experiences": experiences},
            mode="chat", task="life_plan",
        )
        selected = next((item for item in experiences if item["id"] == raw.get("experience_id")), None)
        if selected:
            selected_id = selected["id"]
            selected = await engine.procedure_store.load(
                agent.id, selected_id, engine.get_sim_timestamp(),
            )
            reused = self._reuse_experience(agent, engine, selected, intention) if selected else None
            if reused:
                return reused
            if selected:
                await engine.procedure_store.fail(
                    agent.id, selected_id, "当前条件不可用，无法开始经验步骤", engine.get_sim_timestamp(),
                )
        raw_steps = raw.get("steps") if isinstance(raw.get("steps"), list) else []
        steps = []
        for item in raw_steps[:6]:
            if not isinstance(item, dict) or not isinstance(item.get("action"), dict):
                continue
            description = str(item.get("description") or item["action"].get("content") or "").strip()
            if not description:
                continue
            action = dict(item["action"])
            action["content"] = description
            action["source"] = "life_plan"
            action["supports_goal_ids"] = [str(value) for value in action.get("supports_goal_ids", raw.get("goal_ids", []))]
            steps.append(PlanStep(
                description=description[:160], action=action,
                expected_outcome=[value if isinstance(value, dict) else {"description": str(value)}
                                  for value in item.get("expected_outcome", [])],
            ))
        if not steps:
            raise RuntimeError("life planner returned no executable steps")
        review_after = max(15, min(180, int(raw.get("review_after_minutes", 60))))
        return LifePlan(
            focus=str(raw.get("focus", steps[0].description))[:180],
            motive=str(raw.get("motive", ""))[:240],
            created_at=engine.get_sim_timestamp(),
            review_at=engine.get_sim_timestamp() + review_after,
            goal_ids=[str(value) for value in raw.get("goal_ids", [])],
            steps=steps,
            replan_reason=replan_reason[:240],
            purpose=intention["purpose"], start_location=agent.state.current_location,
        )

    @staticmethod
    def _reuse_experience(agent, engine, experience: dict, intention: dict) -> LifePlan | None:
        steps = [PlanStep(description=action.get("content", ""), action=dict(action))
                 for action in experience["actions"]]
        if not steps or not engine.interactions.validate(agent, steps[0].action, engine).feasible:
            return None
        active_goals = {goal.id for goal in agent.mental_state.goals if goal.status == "active"}
        if any(step.action.get("goal_id") and step.action["goal_id"] not in active_goals for step in steps):
            return None
        now = engine.get_sim_timestamp()
        return LifePlan(
            focus=experience["focus"], motive="沿用成功经验", created_at=now,
            review_at=now + 60, steps=steps, purpose=intention["purpose"],
            experience_id=experience["id"], start_location=agent.state.current_location,
        )

    def next_plan_action(self, agent, engine) -> dict:
        plan = agent.mental_state.life_plan
        if not plan or plan.status != "active":
            raise RuntimeError("agent has no active life plan")
        step = plan.current_step
        if not step:
            plan.status = "completed"
            raise RuntimeError("life plan is complete")
        action = dict(step.action)
        interaction_type = str(action.get("interaction_type") or action.get("action") or "")
        if interaction_type != "move" and not action.get("location"):
            action["location"] = agent.state.current_location
        elif interaction_type == "move" and not action.get("location"):
            locations = engine.context.world_query.locations_for_intention(
                agent, engine, action, limit=1,
            )
            if locations and locations[0]["id"] != agent.state.current_location:
                action["location"] = locations[0]["id"]
        action.update({
            "plan_id": plan.id,
            "plan_step_id": step.id,
            "predicted_effects": step.expected_outcome,
            "source": "experience" if plan.experience_id else "life_plan",
        })
        result = engine.interactions.validate(agent, action, engine, step.id)
        if not result.feasible:
            raise RuntimeError("plan step blocked: " + "; ".join(result.reasons))
        resolved = result.resolved_action
        resolved["validated_effects"] = result.expected_effects
        resolved["duration_minutes"] = result.estimated_minutes
        resolved["plan_id"] = plan.id
        resolved["plan_step_id"] = step.id
        return resolved

    async def decide(self, agent, engine, context: dict) -> dict:
        """Conflict-only two-stage decision retained outside the normal tick path."""
        if self.llm.fallback:
            raise RuntimeError("cognitive planner unavailable in fallback mode")
        proposal_raw = await self._json_call(
            """你负责提出角色的可审计决策候选，不直接执行。只输出JSON。
严格按以下JSON结构输出，不得把assessment写成字符串：
{"assessment":{"confirmed_fact_ids":[],"belief_ids":[],"uncertainties":[],"active_motives":[],"constraints":[],"relevant_goal_ids":[],"relevant_intention_ids":[]},"options":[{"id":"option_id","action":{"interaction_type":"wait","content":"玩家能看到的自然行动描述","duration_minutes":5},"expected_effects":[],"risks":[],"assumptions":[],"supports_goal_ids":[],"conflicts_with_goal_ids":[]}],"preferred_option_id":"option_id","mental_update":{}}
options必须有2到4项，每项action使用以下interaction_type之一：
move, consume, rest, communicate, inspect, request_service, take, put, transfer, use_resource, operate, produce, work_on_goal, wait。
move必须在action.location中填写known_locations里真实存在的地点ID，不能只在content中写地点名称。
环境操作必须引用上下文中真实存在的resource_id、device_id或process_id；缺少信息时先inspect。
work_on_goal必须在action中给出真实goal_id和0到1之间的progress_delta；已完成目标不得继续推进。
content只描述旁观者能看到的动作，不得出现goal_id、resource_id、device_id、process_id、括号假设或系统校验说明；ID和假设只放结构化字段。
必须检查mental_state.recent_outcomes和last_decision；除非上次结果显示目标仍有实质进度未完成，否则不要换一种说法重复相同行动。
结合当前地点、下一安排的地点和路程；需要异地履行目标时，将move作为真实候选，而不是一直在原地填充活动。
确定事实只能通过confirmed_fact_ids引用；不知道的内容放入uncertainties。
不要输出隐藏思维过程，只输出事实引用、候选、预期后果、风险和假设。""",
            context,
            mode="chat", task="cognitive_proposal",
        )
        proposal = DecisionProposal.from_dict(proposal_raw)
        if not proposal.options:
            raise RuntimeError("planner proposal contains no valid action options")
        proposal.options = proposal.options[:4]

        valid_fact_ids, invalid_fact_ids = engine.fact_ledger.validate_references(
            proposal.assessment.confirmed_fact_ids, agent.id, engine.get_sim_timestamp()
        )
        proposal.assessment.confirmed_fact_ids = valid_fact_ids
        if invalid_fact_ids:
            proposal.assessment.uncertainties.append({
                "question": "部分引用事实不存在、来自未来或当前角色不可知",
                "invalid_fact_ids": invalid_fact_ids,
            })

        validations = []
        option_map = {}
        for option in proposal.options:
            action = dict(option.action)
            action.update({
                "option_id": option.id,
                "predicted_effects": option.expected_effects,
                "assumptions": option.assumptions,
                "supports_goal_ids": option.supports_goal_ids,
                "mental_update": proposal.mental_update,
                "source": "cognitive_planner",
            })
            result = engine.interactions.validate(agent, action, engine, option.id)
            validations.append(result.to_dict())
            option_map[option.id] = result

        feasible_ids = [item.option_id for item in option_map.values() if item.feasible]
        if not feasible_ids:
            reasons = [reason for item in validations for reason in item.get("reasons", [])]
            raise RuntimeError("planner returned no feasible option: " + "; ".join(reasons))

        selection = await self._json_call(
            """你是同一角色。系统已经验证了候选行动。只从feasible=true的候选ID中选一个。
严格输出JSON对象：{"selected_option_id":"可行候选ID","brief_reason":"简短依据","mental_update":{}}。
brief_reason只能引用给出的事实、目标、风险和验证结果，不添加新事实。
mobility.travel_required为true且已超过latest_departure时，除非存在更高优先级的已确认约束，否则必须选择可行的move候选，不能选择等待、观察、休息或准备类填充行动。""", 
            {
                "character": agent.persona_text(),
                "assessment": asdict(proposal.assessment),
                "options": [asdict(option) for option in proposal.options],
                "validations": validations,
                "preferred_option_id": proposal.preferred_option_id,
                "mobility": context.get("mobility"),
                "needs": context.get("needs"),
            },
            mode="chat", task="cognitive_selection",
        )
        selected_id = str(selection.get("selected_option_id", ""))
        if selected_id not in feasible_ids:
            selected_id = proposal.preferred_option_id if proposal.preferred_option_id in feasible_ids else feasible_ids[0]
        chosen = option_map[selected_id].resolved_action
        chosen["validated_effects"] = option_map[selected_id].expected_effects
        chosen["duration_minutes"] = option_map[selected_id].estimated_minutes
        chosen["mental_update"] = selection.get("mental_update") if isinstance(selection.get("mental_update"), dict) else proposal.mental_update
        chosen["decision_reason"] = str(selection.get("brief_reason", ""))[:240]
        chosen["planning"] = {"proposal": asdict(proposal), "validations": validations}
        agent.mental_state.last_decision = {
            "selected_option_id": selected_id,
            "brief_reason": chosen["decision_reason"],
            "confirmed_fact_ids": valid_fact_ids,
            "uncertainties": proposal.assessment.uncertainties,
        }
        return chosen

    async def _json_call(self, system: str, payload: dict,
                         mode: str = "chat", task: str = "json_call") -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        data = None
        budget = 4096
        for budget in (4096, 8192):
            data = await self.llm._chat_raw(
                messages, temperature=0.35, max_tokens=budget,
                response_format={"type": "json_object"}, mode=mode, task=task,
            )
            choice = data.get("choices", [{}])[0]
            if str(choice.get("finish_reason", "")) != "length":
                break
        choice = (data or {}).get("choices", [{}])[0]
        finish_reason = str(choice.get("finish_reason", ""))
        content = choice.get("message", {}).get("content") or ""
        if finish_reason == "length":
            raise RuntimeError(f"planner response exceeded the {budget}-token output budget")
        if not content.strip():
            raise RuntimeError(f"planner returned empty content (finish_reason={finish_reason or 'unknown'})")
        try:
            parsed = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(f"planner returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("planner response must be a JSON object")
        return parsed
