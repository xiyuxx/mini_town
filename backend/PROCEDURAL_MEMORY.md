# Resident procedural memory

Successful multi-step life plans are stored as resident-owned procedures in SQLite.
They reuse the existing `LifePlan` executor and per-step feasibility checks.

## Discovery and loading

1. Filter descriptors by resident, simulation time, starting location, active goals,
   location permissions, referenced resources/processes, archival and cooldown state.
2. Rank eligible descriptors by semantic or lexical similarity, reliability and recency.
3. Send at most four summaries to the existing planning call. Summaries contain ID,
   purpose, focus, action types and success/failure counts, never step bodies or vectors.
4. Accept only an ID from those candidates, then load its steps with owner/time checks.
5. Create fresh plan/step IDs, validate the first step and revalidate each subsequent step.

Local token-hash vectors support offline retrieval. Configured embedding providers build
semantic vectors in background tasks; discovery falls back to lexical similarity if the
provider fails. Model identifiers guard against comparing incompatible vectors.
Descriptors are streamed from SQLite and ranking retains a bounded candidate list. Query
work remains linear in eligible descriptors; a vector index is deferred until measured
latency warrants it. There is no twelve-procedure knowledge cap.

## Learning and maintenance

Only fully completed plans with at least two steps and a substantive activity are learned.
Execution metadata and predicted/actual effects are excluded from stored actions.
Structural fingerprints ignore prose and timing differences while preserving action order,
objects, quantities and other executable arguments. Equivalent methods reinforce the same
record; different methods remain separate.

Temporary shortages or unavailable services cool a procedure down for 60 simulated minutes.
Other failures cool it down for 180 minutes; three structural failures archive it.
These categories currently use the executor's reason text, rather than structured error codes.
An independently successful execution of the same method reactivates it.

Existing `experiences` in cognition snapshots migrate transactionally into the procedure table
and are removed from the snapshot, so repeated startup does not double-count successes.
New snapshots contain only the active plan and its procedure reference. Reset drains background
work before clearing the procedure table.

## Current boundaries

Procedures remain tied to the original starting location and concrete world IDs; parameter
binding and learning through teaching are future extensions. Discovery checks object existence,
not the present feasibility of every future step: earlier steps can establish later prerequisites.
Reliability totals are current store state, not a complete time-travel history. Records updated
beyond the current simulation time are withheld rather than replaying historical statistics.

Verification covers 1,001 stored procedures, bounded summaries, retrieval of an old relevant
method, structural deduplication, ownership/time boundaries, migration idempotence, embedding
fallback, cooldown/archive/reactivation, restart recovery and reset with pending indexing.
