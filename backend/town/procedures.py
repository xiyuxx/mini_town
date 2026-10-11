"""Resident-owned procedural memory with bounded discovery and lazy loading."""

import hashlib
import json
import os
import uuid

import aiosqlite

from .embedding import EmbeddingProvider
from .world import can_enter


ACTION_FIELDS = {
    "interaction_type", "location", "target", "target_agent_id", "resource_id",
    "device_id", "process_id", "quantity", "resource_kind", "duration_minutes",
    "goal_id", "progress_delta", "supports_goal_ids", "quality", "anchor_id",
    "target_anchor_id", "destination", "container_id", "destination_container_id",
    "capability", "product_kind", "product_name", "properties", "target_id",
}


class ProcedureStore:
    def __init__(self, db_path):
        self.db_path = db_path
        self.provider = None

    async def init_db(self):
        migrated = []
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                start_location TEXT NOT NULL, purpose TEXT NOT NULL, focus TEXT NOT NULL,
                actions TEXT NOT NULL, embedding TEXT NOT NULL,
                semantic_embedding TEXT NOT NULL DEFAULT '[]', embedding_model TEXT NOT NULL DEFAULT '',
                learned_at INTEGER NOT NULL,
                completed_at INTEGER NOT NULL, successes INTEGER NOT NULL DEFAULT 1,
                failures INTEGER NOT NULL DEFAULT 0, cooldown_until INTEGER NOT NULL DEFAULT 0,
                structural_failures INTEGER NOT NULL DEFAULT 0, conditions TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0, last_failure TEXT NOT NULL DEFAULT '',
                UNIQUE(agent_id, fingerprint))""")
            await db.execute("CREATE INDEX IF NOT EXISTS procedure_discovery ON procedures "
                             "(agent_id, start_location, archived, learned_at)")
            # Migrate the old snapshot list in the same transaction as its removal.
            table = await db.execute("SELECT name FROM sqlite_master WHERE name='cognition_snapshots'")
            if await table.fetchone():
                cursor = await db.execute("SELECT agent_id, state_json FROM cognition_snapshots")
                for agent_id, state_json in await cursor.fetchall():
                    try:
                        payload = json.loads(state_json)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    legacy = payload.pop("experiences", None)
                    if legacy is not None:
                        for item in legacy:
                            migrated.append(await self.learn(agent_id, item, db=db))
                        await db.execute("UPDATE cognition_snapshots SET state_json=? WHERE agent_id=?",
                                         (json.dumps(payload, ensure_ascii=False), agent_id))
            await db.commit()
        return migrated

    async def learn(self, agent_id, experience, db=None):
        if db is None:
            async with aiosqlite.connect(self.db_path) as connection:
                key = await self.learn(agent_id, experience, db=connection)
                await connection.commit()
                return key
        actions = [{key: value for key, value in action.items()
                    if key in ACTION_FIELDS or key == "content"}
                   for action in experience["actions"]]
        structure = [{key: value for key, value in action.items()
                      if key in ACTION_FIELDS and key != "duration_minutes"}
                     for action in actions]
        fingerprint = hashlib.sha256(json.dumps(
            [experience["start_location"], structure], sort_keys=True,
            ensure_ascii=False, separators=(",", ":"),
        ).encode()).hexdigest()
        conditions = {
            "goal_ids": sorted({action["goal_id"] for action in actions if action.get("goal_id")}),
            "locations": sorted({action["location"] for action in actions if action.get("location")}),
            "resource_ids": sorted({action[key] for action in actions for key in ("resource_id", "device_id")
                                    if action.get(key)}),
            "process_ids": sorted({action["process_id"] for action in actions if action.get("process_id")}),
            "action_types": list(dict.fromkeys(action.get("interaction_type", "wait") for action in actions)),
        }
        text = experience["purpose"] + " " + experience["focus"]
        vector = EmbeddingProvider._token_hash_embed(text, dims=128)
        now = experience["completed_at"]
        await db.execute("""INSERT INTO procedures
                (id, agent_id, fingerprint, start_location, purpose, focus, actions,
                 embedding, learned_at, completed_at, successes, conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, fingerprint) DO UPDATE SET
                completed_at=MAX(completed_at, excluded.completed_at),
                successes=successes + excluded.successes, archived=0, cooldown_until=0,
                structural_failures=0, last_failure=''""",
                (experience.get("id") or f"procedure_{uuid.uuid4().hex[:12]}", agent_id,
                 fingerprint, experience["start_location"], experience["purpose"][:240],
                 experience["focus"][:240], json.dumps(actions, ensure_ascii=False),
                 json.dumps(vector), now, now, experience.get("successes", 1), json.dumps(conditions)))
        cursor = await db.execute("SELECT id FROM procedures WHERE agent_id=? AND fingerprint=?",
                                  (agent_id, fingerprint))
        return (await cursor.fetchone())[0]

    async def index(self, key):
        """Build optional semantic vectors outside the simulation tick."""
        if self.provider is None or self.provider.fallback:
            return
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT purpose, focus FROM procedures WHERE id=? "
                                      "AND embedding_model!=?", (key, self.provider.model))
            row = await cursor.fetchone()
        if row:
            vector = await self.provider.embed_one(" ".join(row))
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("UPDATE procedures SET semantic_embedding=?, embedding_model=? WHERE id=?",
                                 (json.dumps(vector), self.provider.model, key))
                await db.commit()

    async def discover(self, agent, engine, intention, limit=4):
        """Scan lightweight descriptors only; no step bodies enter the prompt."""
        now = engine.get_sim_timestamp()
        query = EmbeddingProvider._token_hash_embed(intention.get("purpose", ""), dims=128)
        semantic_query = None
        if self.provider is not None and not self.provider.fallback:
            try:
                semantic_query = await self.provider.embed_one(intention.get("purpose", ""))
            except Exception:
                pass
        ranked = []
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""SELECT id, purpose, focus, embedding, successes,
                failures, completed_at, semantic_embedding, embedding_model, conditions FROM procedures WHERE agent_id=? AND start_location=?
                AND learned_at<=? AND completed_at<=? AND archived=0 AND cooldown_until<=?""",
                (agent.id, agent.state.current_location, now, now, now))
            goals = {goal.id for goal in agent.mental_state.goals if goal.status == "active"}
            async for key, purpose, focus, embedding, successes, failures, completed, semantic, model, conditions in cursor:
                conditions = json.loads(conditions)
                if not set(conditions["goal_ids"]) <= goals:
                    continue
                if any(not can_enter(agent.id, location) for location in conditions["locations"]):
                    continue
                if any(engine.resources.get(key) is None for key in conditions["resource_ids"]):
                    continue
                if any(key not in engine.resources.processes for key in conditions["process_ids"]):
                    continue
                relevance = EmbeddingProvider.cosine_similarity(query, json.loads(embedding))
                vector = json.loads(semantic)
                if semantic_query and self.provider.model == model and len(semantic_query) == len(vector):
                    relevance = max(relevance, EmbeddingProvider.cosine_similarity(semantic_query, vector))
                if relevance < 0.08:
                    continue
                reliability = (successes + 1) / (successes + failures + 2)
                score = relevance + 0.12 * reliability + 0.03 / (1 + max(0, now - completed) / 1440)
                ranked.append((score, {"id": key, "purpose": purpose, "focus": focus,
                                       "successes": successes, "failures": failures,
                                       "action_types": conditions["action_types"]}))
                # Bounded working memory even when the resident has thousands of skills.
                if len(ranked) > limit * 2:
                    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
                    del ranked[limit:]
        return [item for _, item in sorted(ranked, key=lambda item: (-item[0], item[1]["id"]))[:limit]]

    async def load(self, agent_id, procedure_id, now):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""SELECT focus, actions FROM procedures
                WHERE id=? AND agent_id=? AND learned_at<=? AND completed_at<=? AND archived=0 AND cooldown_until<=?""",
                (procedure_id, agent_id, now, now, now))
            row = await cursor.fetchone()
        return {"id": procedure_id, "focus": row[0], "actions": json.loads(row[1])} if row else None

    async def fail(self, agent_id, procedure_id, reason, now):
        # Temporary world conditions cool down; repeated structural failures archive.
        temporary = any(word in reason for word in ("不足", "不可用", "无人", "未开放", "状态", "空闲"))
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""UPDATE procedures SET failures=failures+1, last_failure=?,
                cooldown_until=?, structural_failures=structural_failures + ?,
                archived=CASE WHEN ?=1 AND structural_failures>=2 THEN 1 ELSE archived END
                WHERE id=? AND agent_id=?""",
                (reason[:240], now + (60 if temporary else 180), int(not temporary), int(not temporary),
                 procedure_id, agent_id))
            await db.commit()

    async def clear(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM procedures")
            await db.commit()
