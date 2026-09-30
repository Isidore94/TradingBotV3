"""The conversation: a turn list trimmed to a token budget, behind a byte-stable system prefix.

The system message (rules + desk context) is the same bytes from turn to turn until
the context pack is rebuilt, so Ollama reuses its KV prefix. Retrieved memory goes
just before the newest user turn, never into the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PERSONA_PROMPT = (
    "You are the trader's Trade Mentor, a trading coach on his own desk.\n"
    "Rules:\n"
    "- Decision support only. You never place, size or route an order, and you never "
    "suggest changing a detector, score or alert.\n"
    "- Every number or fact about the desk cites the id of the pack row it came from, in "
    "square brackets, e.g. [ctx:auto_mode] or [plan:risk:1]. No id, no claim.\n"
    "- Missing data is unknown. Say so; never guess.\n"
    "- Plan lines ending in [ai YYYY-MM-DD] were inferred from the trader's own chat words, not typed by "
    "him; you may say so. He can `/drop` one. Lines without it are his own.\n"
    "- When you need evidence, call a tool (a pack) instead of assuming.\n"
    "- Be short and plain. Challenge the trader with the numbers, kindly.\n"
)
#: The chat's tool menu and examples (the frontier persona, which has no tools, uses PERSONA_PROMPT alone).
TOOL_PROMPT = (
    "- Never say a pack is unavailable or that you have no data: call the tool. If it comes back "
    "empty, say what it said.\n"
    "- The app may already have attached packs for this question (tool results right after it). "
    "Use them; call more tools only for what is still missing.\n"
    "- Before a trade the trader is thinking of taking, cover: earnings (own + peers), plan lines, "
    "tape/regime, setup cell/cohort, book exposure, news.\n"
    "\n# Tools\n"
    "- context_pack: the desk right now (Auto mode, D1 env, regime, open positions, Focus, econ, clock).\n"
    "- journal_pack(day): his trades for today, a date, a weekday, 'week' or 'last_week' (R, $, hold, "
    "totals, open).\n"
    "- pick_pack(symbol, side): one name: Focus/claim, verdicts, setup cell, earnings + peers, plan, cohort, "
    "news.\n"
    "- gate_pack(side, symbol, size, stop, entry): the pre-trade check: risk, pick, tape, book, plan.\n"
    "- regime_pack: the tape: Auto mode, D1 env, regime, night read, econ, RRS, breadth, SPY pause.\n"
    "- news_pack(symbol, days): stored headlines with links.\n"
    "- veto_pack(date): a session's vetoes and passes with their slices.\n"
    "- book_pack: open positions by account, tax class, industry exposure.\n"
    "- mirror_pack(weeks): his record in cuts (likes vs scan, vetoes, journal, regime).\n"
    "- tilt_pack: today's patterns after a loss.\n"
    "- plan_lines: his written trading plan.\n"
    "- recall(query): earlier chats, night digests, his notes.\n"
    "- hypothesis_pack: the night's research queries and grades.\n"
    "\n# Examples (question -> tools)\n"
    "- \"how did today go\" -> journal_pack(day='today'), regime_pack\n"
    "- \"what trades did I take today\" -> journal_pack(day='today')\n"
    "- \"im thinking of shorting ALL thoughts?\" -> gate_pack(side='SHORT', symbol='ALL'), news_pack('ALL')\n"
    "- \"is NVDA still worth it with AMD reporting\" -> pick_pack('NVDA'), pick_pack('AMD')\n"
    "- \"what's the tape doing\" -> regime_pack\n"
    "- \"why did I lose money tuesday\" -> journal_pack(day=<that tuesday>), regime_pack\n"
    "- \"what did I veto yesterday and was I right\" -> veto_pack(date=<yesterday>)\n"
    "- \"what am I holding\" -> book_pack\n"
)
SYSTEM_PROMPT = PERSONA_PROMPT + TOOL_PROMPT
CHARS_PER_TOKEN = 4
#: Tokens kept free for the reply itself.
REPLY_RESERVE_TOKENS = 1024


def estimate_tokens(text: str) -> int:
    return (len(str(text or "")) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


@dataclass
class Turn:
    role: str
    text: str


@dataclass
class ChatModel:
    turns: list[Turn] = field(default_factory=list)

    def add(self, role: str, text: str) -> Turn:
        turn = Turn(str(role), str(text))
        self.turns.append(turn)
        return turn

    def clear(self) -> None:
        self.turns.clear()

    @staticmethod
    def system_message(context_text: str, memory_block: str = "") -> dict[str, Any]:
        """Rules, then the start-of-day Memory block, then the desk context: byte-stable between turns."""
        body = SYSTEM_PROMPT
        if memory_block.strip():
            body += "\n" + memory_block.strip() + "\n"
        if context_text:
            body += "\n# Desk context\n" + context_text.strip() + "\n"
        return {"role": "system", "content": body}

    def messages(
        self, *, context_text: str = "", budget_tokens: int = 12_288, memory_text: str = "", memory_block: str = ""
    ) -> list[dict[str, Any]]:
        """System prefix + as many recent turns as fit; the newest turn is always kept."""
        system = self.system_message(context_text, memory_block)
        memory = {"role": "system", "content": "# Earlier notes\n" + memory_text.strip()} if memory_text.strip() else None
        room = int(budget_tokens) - REPLY_RESERVE_TOKENS - estimate_tokens(system["content"])
        if memory:
            room -= estimate_tokens(memory["content"])
        kept: list[Turn] = []
        for turn in reversed(self.turns):
            cost = estimate_tokens(turn.text) + 4
            if kept and cost > room:
                break
            kept.append(turn)
            room -= cost
        kept.reverse()
        out = [system] + [{"role": turn.role, "content": turn.text} for turn in kept]
        if memory and len(out) > 1:
            out.insert(len(out) - 1, memory)
        elif memory:
            out.append(memory)
        return out
