"""Candidate registry with provenance, leases, and lifecycle (plan.md Packet D).

One store replaces the competing watchlist writers (open scan replace,
30-minute auto-populate rotation, near-extreme appends, VWAP removals): every
automation is a *source* holding a membership lease on a symbol/side, and the
bot-facing scan pool is DERIVED from the registry. Text watchlists become
imports/exports, never the coordination database.

Rules enforced here:
- user-owned entries survive every automation pass (only an explicit user
  action removes them);
- one symbol/side is one candidate no matter how many sources add it;
- removing a source keeps the candidate alive while other sources remain;
- one lifecycle transition produces exactly one typed event (deduped);
- persistence is atomic and version-guarded so a stale writer cannot erase a
  newer state (optimistic concurrency);
- a restart reconstructs identical active candidates from disk.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

REGISTRY_VERSION = "candidate_registry_v1"

SOURCE_USER = "user"

# sec 16.6 live-pool priority: earlier is more important.
DEFAULT_SOURCE_PRIORITY = (
    SOURCE_USER,
    "focus",
    "master_setup",
    "pullback_defiance",
    "open_scan",
    "near_extreme",
    "auto_populate",
    "outcome_required",
    "background",
)

STAGES = (
    "DEVELOPING",
    "DEFIANT",
    "HOLDING",
    "READY",
    "TRIGGERED",
    "INVALID",
    "EXPIRED",
)

_STAGE_POOL_PRIORITY = {
    "TRIGGERED": 0,
    "READY": 1,
    "DEFIANT": 2,
    "HOLDING": 3,
    "DEVELOPING": 4,
}


class StaleWriterError(RuntimeError):
    """The on-disk registry advanced past this writer's loaded generation."""


@dataclass
class Membership:
    source: str
    added_at: str
    lease_expires_at: str | None = None
    note: str = ""


@dataclass
class TransitionEvent:
    symbol: str
    side: str
    from_stage: str
    to_stage: str
    ts: str


@dataclass
class Candidate:
    symbol: str
    side: str
    stage: str = "DEVELOPING"
    rank_score: float = 0.0
    memberships: dict[str, Membership] = field(default_factory=dict)
    stage_history: list[list[str]] = field(default_factory=list)  # [stage, ts]

    @property
    def active(self) -> bool:
        return bool(self.memberships) and self.stage not in ("EXPIRED",)


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now()).isoformat(timespec="seconds")


class CandidateRegistry:
    def __init__(self, source_priority: tuple[str, ...] = DEFAULT_SOURCE_PRIORITY) -> None:
        self.source_priority = source_priority
        self._candidates: dict[tuple[str, str], Candidate] = {}
        self._generation = 0            # bumps on every mutation
        self._loaded_generation = 0     # what we last read from disk

    # ------------------------------------------------------------------
    # membership / provenance
    # ------------------------------------------------------------------
    def add(
        self,
        symbol: str,
        side: str,
        source: str,
        *,
        lease_minutes: int | None = None,
        note: str = "",
        rank_score: float | None = None,
        now: datetime | None = None,
    ) -> Candidate:
        symbol = symbol.strip().upper()
        side = side.strip().upper()
        key = (symbol, side)
        candidate = self._candidates.get(key)
        if candidate is None:
            candidate = Candidate(symbol=symbol, side=side)
            self._candidates[key] = candidate
        lease = None
        if lease_minutes is not None:
            lease = _now_iso((now or datetime.now()) + timedelta(minutes=lease_minutes))
        candidate.memberships[source] = Membership(
            source=source,
            added_at=_now_iso(now),
            lease_expires_at=lease,
            note=note,
        )
        if rank_score is not None:
            candidate.rank_score = float(rank_score)
        if candidate.stage == "EXPIRED":
            candidate.stage = "DEVELOPING"
        self._generation += 1
        return candidate

    def remove_source(self, symbol: str, side: str, source: str, *, actor: str = "automation") -> bool:
        """Remove one source's membership. Only a user actor may remove the
        user source - automation can never erase hand-entered names."""
        if source == SOURCE_USER and actor != "user":
            return False
        candidate = self._candidates.get((symbol.strip().upper(), side.strip().upper()))
        if candidate is None or source not in candidate.memberships:
            return False
        del candidate.memberships[source]
        self._generation += 1
        return True

    def sync_source(
        self,
        source: str,
        symbols: dict[str, list[str]],
        *,
        lease_minutes: int | None = None,
        now: datetime | None = None,
    ) -> None:
        """Replace one automation source's memberships with a new set.

        `symbols` maps side -> list of symbols. Other sources (and therefore
        user entries) are untouched: this is how the open scan or the
        30-minute auto-populate rotates its own names without competing.
        """
        if source == SOURCE_USER:
            raise ValueError("sync_source is for automation sources; use import_user_watchlist")
        desired = {
            (sym.strip().upper(), side.strip().upper())
            for side, syms in symbols.items()
            for sym in syms
        }
        for key, candidate in list(self._candidates.items()):
            if source in candidate.memberships and key not in desired:
                del candidate.memberships[source]
                self._generation += 1
        for sym, side in desired:
            self.add(sym, side, source, lease_minutes=lease_minutes, now=now)

    def import_user_watchlist(self, side: str, symbols: list[str], *, now: datetime | None = None) -> None:
        """Sync the user source to a hand-edited list (a user action)."""
        side = side.strip().upper()
        desired = {s.strip().upper() for s in symbols if s.strip()}
        for (sym, cand_side), candidate in list(self._candidates.items()):
            if cand_side == side and SOURCE_USER in candidate.memberships and sym not in desired:
                del candidate.memberships[SOURCE_USER]
                self._generation += 1
        for sym in desired:
            self.add(sym, side, SOURCE_USER, now=now)

    def expire_leases(self, now: datetime | None = None) -> list[tuple[str, str, str]]:
        """Drop expired automation leases; user entries never expire."""
        moment = now or datetime.now()
        removed: list[tuple[str, str, str]] = []
        for (symbol, side), candidate in self._candidates.items():
            for source, membership in list(candidate.memberships.items()):
                if source == SOURCE_USER or not membership.lease_expires_at:
                    continue
                if datetime.fromisoformat(membership.lease_expires_at) <= moment:
                    del candidate.memberships[source]
                    removed.append((symbol, side, source))
                    self._generation += 1
        return removed

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def set_stage(
        self,
        symbol: str,
        side: str,
        stage: str,
        *,
        now: datetime | None = None,
    ) -> TransitionEvent | None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        candidate = self._candidates.get((symbol.strip().upper(), side.strip().upper()))
        if candidate is None or candidate.stage == stage:
            return None  # one lifecycle transition -> one event; repeats dedupe
        event = TransitionEvent(
            symbol=candidate.symbol,
            side=candidate.side,
            from_stage=candidate.stage,
            to_stage=stage,
            ts=_now_iso(now),
        )
        candidate.stage = stage
        candidate.stage_history.append([stage, event.ts])
        self._generation += 1
        return event

    # ------------------------------------------------------------------
    # views
    # ------------------------------------------------------------------
    def get(self, symbol: str, side: str) -> Candidate | None:
        return self._candidates.get((symbol.strip().upper(), side.strip().upper()))

    def active_candidates(self, side: str | None = None) -> list[Candidate]:
        out = [
            c
            for c in self._candidates.values()
            if c.active and (side is None or c.side == side.strip().upper())
        ]
        return sorted(out, key=lambda c: (c.side, c.symbol))

    def derive_live_pool(self, side: str, max_size: int) -> list[str]:
        """Priority-ordered live scan pool (sec 16.6); never mutates inputs."""

        def source_rank(candidate: Candidate) -> int:
            ranks = [
                self.source_priority.index(s)
                for s in candidate.memberships
                if s in self.source_priority
            ]
            return min(ranks) if ranks else len(self.source_priority)

        eligible = [
            c
            for c in self.active_candidates(side)
            if c.stage not in ("INVALID", "EXPIRED")
        ]
        eligible.sort(
            key=lambda c: (
                _STAGE_POOL_PRIORITY.get(c.stage, 9),
                source_rank(c),
                -c.rank_score,
                c.symbol,
            )
        )
        return [c.symbol for c in eligible[: max(0, int(max_size))]]

    # ------------------------------------------------------------------
    # persistence (atomic + optimistic concurrency)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": REGISTRY_VERSION,
            "generation": self._generation,
            "candidates": [
                {
                    "symbol": c.symbol,
                    "side": c.side,
                    "stage": c.stage,
                    "rank_score": c.rank_score,
                    "stage_history": c.stage_history,
                    "memberships": {s: asdict(m) for s, m in c.memberships.items()},
                }
                for c in self._candidates.values()
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CandidateRegistry":
        registry = cls()
        registry._generation = int(payload.get("generation", 0))
        registry._loaded_generation = registry._generation
        for row in payload.get("candidates", []):
            candidate = Candidate(
                symbol=row["symbol"],
                side=row["side"],
                stage=row.get("stage", "DEVELOPING"),
                rank_score=float(row.get("rank_score", 0.0)),
                stage_history=[list(x) for x in row.get("stage_history", [])],
                memberships={
                    s: Membership(**m) for s, m in row.get("memberships", {}).items()
                },
            )
            registry._candidates[(candidate.symbol, candidate.side)] = candidate
        return registry

    def save(self, path: Path | str, *, force: bool = False) -> None:
        path = Path(path)
        if not force and path.exists():
            try:
                on_disk = json.loads(path.read_text(encoding="utf-8"))
                disk_generation = int(on_disk.get("generation", 0))
            except (json.JSONDecodeError, OSError, ValueError):
                disk_generation = 0
            if disk_generation > max(self._loaded_generation, self._generation):
                raise StaleWriterError(
                    f"registry on disk is at generation {disk_generation}, "
                    f"this writer loaded {self._loaded_generation}; reload and merge"
                )
        payload = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=1)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
        self._loaded_generation = self._generation

    @classmethod
    def load(cls, path: Path | str) -> "CandidateRegistry":
        path = Path(path)
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    # ------------------------------------------------------------------
    # exports
    # ------------------------------------------------------------------
    def export_watchlist_lines(self, side: str) -> list[str]:
        """Compatibility text-file view of the active pool (a derived export)."""
        return [c.symbol for c in self.active_candidates(side)]
