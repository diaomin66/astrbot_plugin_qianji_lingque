from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


MODE_ALIASES = {
    "quiet": "quiet",
    "安静": "quiet",
    "normal": "normal",
    "普通": "normal",
    "active": "active",
    "积极": "active",
}

MODE_LABELS = {
    "quiet": "安静",
    "normal": "普通",
    "active": "积极",
}


@dataclass
class PluginConfig:
    enabled: bool = True
    enabled_groups: list[str] = field(default_factory=list)
    disabled_groups: list[str] = field(default_factory=list)
    mode: str = "normal"
    group_modes: dict[str, str] = field(default_factory=dict)
    max_context_messages: int = 24
    merge_window_seconds: float = 1.2
    min_reply_interval_seconds: float = 18.0
    bot_cooldown_after_reply_seconds: float = 35.0
    reply_timeout_seconds: float = 12.0
    llm_gate_enabled: bool = True
    llm_gate_timeout_seconds: float = 1.5
    llm_gate_fallback_score: float = 0.4
    max_tracked_groups: int = 200
    group_ttl_seconds: float = 86400.0
    score_threshold_reply: float = 0.78
    score_threshold_ignore: float = 0.35
    debug_explain_enabled: bool = True
    log_decisions_enabled: bool = True
    log_message_excerpt_enabled: bool = False
    takeover_explicit_mentions: bool = True
    bot_aliases: list[str] = field(default_factory=list)
    source: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def from_astrbot_config(cls, config: Any) -> "PluginConfig":
        data = dict(config or {})
        return cls(
            enabled=_bool(data.get("enabled"), True),
            enabled_groups=_string_list(data.get("enabled_groups")),
            disabled_groups=_string_list(data.get("disabled_groups")),
            mode=normalize_mode(str(data.get("mode", "normal"))),
            group_modes=_mode_dict(data.get("group_modes")),
            max_context_messages=_clamp_int(data.get("max_context_messages"), 24, 4, 100),
            merge_window_seconds=_clamp_float(data.get("merge_window_seconds"), 1.2, 0.0, 10.0),
            min_reply_interval_seconds=_clamp_float(
                data.get("min_reply_interval_seconds"),
                18.0,
                0.0,
                600.0,
            ),
            bot_cooldown_after_reply_seconds=_clamp_float(
                data.get("bot_cooldown_after_reply_seconds"),
                35.0,
                0.0,
                1200.0,
            ),
            reply_timeout_seconds=_clamp_float(data.get("reply_timeout_seconds"), 12.0, 1.0, 60.0),
            llm_gate_enabled=_bool(data.get("llm_gate_enabled"), True),
            llm_gate_timeout_seconds=_clamp_float(
                data.get("llm_gate_timeout_seconds"),
                1.5,
                0.2,
                10.0,
            ),
            llm_gate_fallback_score=_clamp_float(
                data.get("llm_gate_fallback_score"),
                0.4,
                0.0,
                1.0,
            ),
            max_tracked_groups=_clamp_int(data.get("max_tracked_groups"), 200, 1, 2000),
            group_ttl_seconds=_clamp_float(data.get("group_ttl_seconds"), 86400.0, 60.0, 604800.0),
            score_threshold_reply=_clamp_float(data.get("score_threshold_reply"), 0.78, 0.0, 1.0),
            score_threshold_ignore=_clamp_float(data.get("score_threshold_ignore"), 0.35, 0.0, 1.0),
            debug_explain_enabled=_bool(data.get("debug_explain_enabled"), True),
            log_decisions_enabled=_bool(data.get("log_decisions_enabled"), True),
            log_message_excerpt_enabled=_bool(data.get("log_message_excerpt_enabled"), False),
            takeover_explicit_mentions=_bool(data.get("takeover_explicit_mentions"), True),
            bot_aliases=_string_list(data.get("bot_aliases"), []),
            source=config,
        ).normalized()

    def normalized(self) -> "PluginConfig":
        if self.score_threshold_ignore > self.score_threshold_reply:
            self.score_threshold_ignore, self.score_threshold_reply = (
                self.score_threshold_reply,
                self.score_threshold_ignore,
            )
        return self

    def save(self) -> None:
        if self.source is None:
            return
        self.source["enabled"] = self.enabled
        self.source["enabled_groups"] = list(self.enabled_groups)
        self.source["disabled_groups"] = list(self.disabled_groups)
        self.source["mode"] = mode_label(self.mode)
        self.source["group_modes"] = _mode_entries(self.group_modes)
        self.source["max_context_messages"] = self.max_context_messages
        self.source["merge_window_seconds"] = self.merge_window_seconds
        self.source["min_reply_interval_seconds"] = self.min_reply_interval_seconds
        self.source["bot_cooldown_after_reply_seconds"] = self.bot_cooldown_after_reply_seconds
        self.source["reply_timeout_seconds"] = self.reply_timeout_seconds
        self.source["llm_gate_enabled"] = self.llm_gate_enabled
        self.source["llm_gate_timeout_seconds"] = self.llm_gate_timeout_seconds
        self.source["llm_gate_fallback_score"] = self.llm_gate_fallback_score
        self.source["max_tracked_groups"] = self.max_tracked_groups
        self.source["group_ttl_seconds"] = self.group_ttl_seconds
        self.source["score_threshold_reply"] = self.score_threshold_reply
        self.source["score_threshold_ignore"] = self.score_threshold_ignore
        self.source["debug_explain_enabled"] = self.debug_explain_enabled
        self.source["log_decisions_enabled"] = self.log_decisions_enabled
        self.source["log_message_excerpt_enabled"] = self.log_message_excerpt_enabled
        self.source["takeover_explicit_mentions"] = self.takeover_explicit_mentions
        self.source["bot_aliases"] = list(self.bot_aliases)
        save_config = getattr(self.source, "save_config", None)
        if callable(save_config):
            save_config()

    def effective_mode(self, group_id: str) -> str:
        return self.group_modes.get(group_id, self.mode)

    def is_group_enabled(self, group_id: str) -> bool:
        if not self.enabled:
            return False
        if group_id in self.disabled_groups:
            return False
        if _all_groups_enabled(self.enabled_groups):
            return True
        if not self.enabled_groups:
            return False
        return group_id in self.enabled_groups

    def enables_all_groups(self) -> bool:
        return _all_groups_enabled(self.enabled_groups)


def normalize_mode(value: str) -> str:
    return parse_mode(value) or "normal"


def parse_mode(value: str) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None
    return MODE_ALIASES.get(stripped.lower(), MODE_ALIASES.get(stripped))


def mode_label(value: str) -> str:
    return MODE_LABELS.get(normalize_mode(value), "普通")


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "开启", "是"}:
            return True
        if normalized in {"0", "false", "no", "off", "关闭", "否"}:
            return False
        return default
    return bool(value)


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    return min(max(_int(value, default), minimum), maximum)


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    return min(max(_float(value, default), minimum), maximum)


def _string_list(value: Any, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else list(default or [])
    try:
        result: list[str] = []
        for item in value:
            stripped = str(item).strip()
            if stripped:
                result.append(stripped)
        return result
    except TypeError:
        return list(default or [])


def _mode_dict(value: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        pairs: list[tuple[str, str]] = []
        for item in value:
            if not isinstance(item, str) or "=" not in item:
                continue
            key, raw_mode = item.split("=", 1)
            pairs.append((key, raw_mode))
        items = pairs
    else:
        return {}
    for key, raw_mode in items:
        group_id = str(key).strip()
        mode = parse_mode(str(raw_mode))
        if group_id and mode:
            result[group_id] = mode
    return result


def _mode_entries(value: dict[str, str]) -> list[str]:
    result: list[str] = []
    for key, raw_mode in value.items():
        group_id = str(key).strip()
        mode = parse_mode(str(raw_mode))
        if group_id and mode:
            result.append(f"{group_id}={mode}")
    return result


def _all_groups_enabled(values: list[str]) -> bool:
    return any(str(value).strip().lower() in {"*", "all", "全部"} for value in values)
