from __future__ import annotations

import re
from typing import Any


TEXT_TAG_PREFIX = "mt_"

COLOR_TEXT = "#C7CDD4"
COLOR_MUTED = "#8E98A5"
COLOR_HEADER = "#D8B4FE"
COLOR_DIVIDER = "#5B6572"
COLOR_LONG = "#7EE787"
COLOR_SHORT = "#FF9A9A"
COLOR_WARN = "#F0C76E"
COLOR_ERROR = "#FF7B72"
COLOR_INFO = "#7DB7FF"
COLOR_TICKER = "#9CDCFE"
COLOR_POSITIVE = "#8FD19E"
COLOR_NEGATIVE = "#F28B82"


HEADER_WORDS = (
    "AVWAP",
    "MASTER",
    "THETA",
    "TRACKER",
    "MARKET PREP",
    "PRIORITY",
    "SETUP",
    "SCENARIO",
    "FACTOR",
    "PLAYBOOK",
    "EARNINGS",
    "CATALYST",
    "NEWS",
    "SEC",
)
WARN_WORDS = ("WARN", "WARNING", "CAUTION", "WATCH", "SOON", "RISK", "MEDIUM")
ERROR_WORDS = ("ERROR", "FAILED", "MISSING", "HIGH RISK", "RESISTANCE")
POSITIVE_WORDS = ("READY", "OK", "SUPPORT", "TARGET", "FAVORITE", "BOUNCE", "BREAKTHROUGH", "CLEAN")


def configure_market_text_tags(widget: Any, *, font_family: str = "Courier New", font_size: int = 10) -> None:
    """Configure shared text tags used by the trading GUI text panes."""
    if widget is None or not hasattr(widget, "tag_configure"):
        return
    widget.tag_configure(f"{TEXT_TAG_PREFIX}header", foreground=COLOR_HEADER, font=(font_family, font_size, "bold"))
    widget.tag_configure(f"{TEXT_TAG_PREFIX}divider", foreground=COLOR_DIVIDER)
    widget.tag_configure(f"{TEXT_TAG_PREFIX}long", foreground=COLOR_LONG, font=(font_family, font_size, "bold"))
    widget.tag_configure(f"{TEXT_TAG_PREFIX}short", foreground=COLOR_SHORT, font=(font_family, font_size, "bold"))
    widget.tag_configure(f"{TEXT_TAG_PREFIX}warn", foreground=COLOR_WARN)
    widget.tag_configure(f"{TEXT_TAG_PREFIX}error", foreground=COLOR_ERROR, font=(font_family, font_size, "bold"))
    widget.tag_configure(f"{TEXT_TAG_PREFIX}info", foreground=COLOR_INFO)
    widget.tag_configure(f"{TEXT_TAG_PREFIX}ticker", foreground=COLOR_TICKER)
    widget.tag_configure(f"{TEXT_TAG_PREFIX}positive", foreground=COLOR_POSITIVE)
    widget.tag_configure(f"{TEXT_TAG_PREFIX}negative", foreground=COLOR_NEGATIVE)
    widget.tag_configure(f"{TEXT_TAG_PREFIX}muted", foreground=COLOR_MUTED)


def set_highlighted_text(
    widget: Any,
    text: str,
    *,
    state_after: str | None = "normal",
    font_family: str = "Courier New",
    font_size: int = 10,
) -> None:
    """Replace widget contents and apply readable full-line tags."""
    if widget is None:
        return
    configure_market_text_tags(widget, font_family=font_family, font_size=font_size)
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    for segment in str(text or "").splitlines(keepends=True) or ([""] if str(text or "") == "" else []):
        line = segment.rstrip("\r\n")
        ending = segment[len(line):]
        tag = line_tag(line)
        if tag:
            widget.insert("end", line + ending, tag)
        else:
            widget.insert("end", line + ending)
    if state_after:
        widget.configure(state=state_after)


def line_tag(line: str) -> str:
    text = str(line or "").strip()
    if not text:
        return ""
    upper = text.upper()
    if _is_divider(text):
        return f"{TEXT_TAG_PREFIX}divider"
    if any(word in upper for word in ERROR_WORDS):
        return f"{TEXT_TAG_PREFIX}error"
    if _looks_short_line(upper):
        return f"{TEXT_TAG_PREFIX}short"
    if _looks_long_line(upper):
        return f"{TEXT_TAG_PREFIX}long"
    if _is_header(text, upper):
        return f"{TEXT_TAG_PREFIX}header"
    if _has_negative_edge(text):
        return f"{TEXT_TAG_PREFIX}negative"
    if _has_positive_edge(text):
        return f"{TEXT_TAG_PREFIX}positive"
    if any(word in upper for word in WARN_WORDS):
        return f"{TEXT_TAG_PREFIX}warn"
    if any(word in upper for word in POSITIVE_WORDS):
        return f"{TEXT_TAG_PREFIX}positive"
    if _looks_symbol_list(text):
        return f"{TEXT_TAG_PREFIX}ticker"
    if text.startswith("[") or text.lower().startswith(("source:", "generated", "updated", "window:")):
        return f"{TEXT_TAG_PREFIX}muted"
    return ""


def configure_treeview_market_tags(tree: Any) -> None:
    if tree is None or not hasattr(tree, "tag_configure"):
        return
    tree.tag_configure("mt_long", foreground=COLOR_LONG)
    tree.tag_configure("mt_short", foreground=COLOR_SHORT)
    tree.tag_configure("mt_positive", foreground=COLOR_POSITIVE)
    tree.tag_configure("mt_negative", foreground=COLOR_NEGATIVE)
    tree.tag_configure("mt_warn", foreground=COLOR_WARN)
    tree.tag_configure("mt_error", foreground=COLOR_ERROR)
    tree.tag_configure("mt_muted", foreground=COLOR_MUTED)


def tree_tags_for_values(values: Any) -> tuple[str, ...]:
    text = " ".join(str(value or "") for value in (values or ())).upper()
    tags: list[str] = []
    if _looks_short_line(text):
        tags.append("mt_short")
    elif _looks_long_line(text):
        tags.append("mt_long")
    if any(word in text for word in ("OPEN", "ACTIVE", "READY", "TARGET", "SUPPORT")):
        tags.append("mt_positive")
    if any(word in text for word in ("CLOSED", "STOP", "FAILED", "ERROR", "MISSING")):
        tags.append("mt_error")
    elif any(word in text for word in ("WATCH", "SOON", "RISK", "MEDIUM")):
        tags.append("mt_warn")
    if _has_negative_edge(text):
        tags.append("mt_negative")
    elif _has_positive_edge(text):
        tags.append("mt_positive")
    return tuple(dict.fromkeys(tags))


def _is_divider(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) >= 8 and set(stripped) <= {"-", "=", "_"}


def _is_header(text: str, upper: str) -> bool:
    if "," in text and _looks_symbol_list(text):
        return False
    if text.startswith("#"):
        return True
    if any(word in upper for word in HEADER_WORDS) and len(text) <= 96 and not text.startswith("- "):
        return True
    return bool(re.fullmatch(r"[A-Z0-9 /&().:_-]{8,}", text)) and len(text) <= 96


def _looks_long_line(upper: str) -> bool:
    return bool(re.search(r"\bLONGS?\b|\bBULLISH\b|\bCALL\b", upper))


def _looks_short_line(upper: str) -> bool:
    return bool(re.search(r"\bSHORTS?\b|\bBEARISH\b|\bPUT\b", upper))


def _has_positive_edge(text: str) -> bool:
    return bool(re.search(r"(?<![\w.])\+\d+(?:\.\d+)?(?:R|%| PTS)?", text, re.IGNORECASE))


def _has_negative_edge(text: str) -> bool:
    return bool(re.search(r"(?<![\w.])-\d+(?:\.\d+)?(?:R|%| PTS)?", text, re.IGNORECASE))


def _looks_symbol_list(text: str) -> bool:
    symbols = re.findall(r"\b[A-Z][A-Z0-9.]{1,5}\b", text)
    return len(symbols) >= 3 and len(text) <= 220
