from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import PluginConfig
from .context import GroupState
from .event_utils import MessageSnapshot

DecisionAction = Literal["reply", "wait", "ignore"]


@dataclass(frozen=True)
class Decision:
    action: DecisionAction
    confidence: float
    reason: str
    is_gray_area: bool = False


class DecisionEngine:
    def decide(self, snapshot: MessageSnapshot, state: GroupState, config: PluginConfig) -> Decision:
        fast_decision = self._fast_gate(snapshot, state, config)
        if fast_decision is not None:
            return fast_decision

        score, reasons = self._score(snapshot, state, config)
        if score >= config.score_threshold_reply:
            return Decision("reply", score, "本地评分足够高：" + "、".join(reasons))
        if score <= config.score_threshold_ignore:
            return Decision("ignore", score, "本地评分较低：" + "、".join(reasons))
        return Decision("wait", score, "本地评分处于灰区，本轮先等待。", is_gray_area=True)

    def _fast_gate(
        self,
        snapshot: MessageSnapshot,
        state: GroupState,
        config: PluginConfig,
    ) -> Decision | None:
        if not config.is_group_enabled(snapshot.group_id):
            return Decision("ignore", 0.0, "插件未在当前群启用。")
        if not snapshot.text:
            return Decision("ignore", 0.0, "空消息不回复。")
        if snapshot.sender_id and snapshot.sender_id == snapshot.self_id:
            return Decision("ignore", 0.0, "忽略 bot 自己的消息。")
        if _looks_like_command(snapshot.text):
            return Decision("ignore", 0.0, "疑似指令消息，不插话。")

        last_user = state.last_user_message()
        if (
            last_user is not None
            and last_user.sender_id == snapshot.sender_id
            and snapshot.timestamp - last_user.timestamp < config.merge_window_seconds
            and not snapshot.is_direct_to_bot
        ):
            return Decision("wait", 0.45, "同一用户正在连续发言，先等待。")
        return None

    def _score(
        self,
        snapshot: MessageSnapshot,
        state: GroupState,
        config: PluginConfig,
    ) -> tuple[float, list[str]]:
        score = 0.08
        reasons = ["基础克制"]

        if snapshot.mentions_bot:
            score += 0.62
            reasons.append("被点名")
        if snapshot.replies_to_bot:
            score += 0.5
            reasons.append("引用 bot")
        if _contains_question(snapshot.text):
            score += 0.22
            reasons.append("问题意图")
        if _contains_help_intent(snapshot.text):
            score += 0.18
            reasons.append("求助意图")

        last_message = state.last_message()
        if last_message is not None and last_message.is_bot and snapshot.sender_id:
            score += 0.1
            reasons.append("接续 bot 发言")

        if len(snapshot.text) <= 2 and not snapshot.is_direct_to_bot:
            score -= 0.18
            reasons.append("短句闲聊")

        recent_user_ids = state.recent_user_ids()
        if snapshot.sender_id:
            recent_user_ids.add(snapshot.sender_id)
        if len(recent_user_ids) >= 2 and not snapshot.is_direct_to_bot:
            score -= 0.14
            reasons.append("群友互聊中")

        if (
            state.last_bot_reply_at
            and snapshot.timestamp - state.last_bot_reply_at < config.min_reply_interval_seconds
            and not snapshot.is_direct_to_bot
        ):
            score -= 0.34
            reasons.append("未过最短回复间隔")

        if (
            state.last_bot_reply_at
            and snapshot.timestamp - state.last_bot_reply_at < config.bot_cooldown_after_reply_seconds
            and not snapshot.is_direct_to_bot
        ):
            score -= 0.24
            reasons.append("bot 刚回复过")

        mode = config.effective_mode(snapshot.group_id)
        if mode == "quiet":
            score -= 0.12
            reasons.append("安静模式")
        elif mode == "active":
            score += 0.12
            reasons.append("积极模式")

        return _clamp(score), reasons


def _looks_like_command(text: str) -> bool:
    stripped = text.strip()
    first_token = stripped.split(" ", 1)[0].lower()
    known_stripped_commands = {
        "help",
        "menu",
        "status",
        "plugin",
        "plugins",
        "帮助",
        "菜单",
        "状态",
        "插件",
        "天气",
        "签到",
        "抽签",
        "运势",
        "读空气",
        "空气",
    }
    return stripped.startswith(("/", "!", "！", "#", "＃", ".")) or first_token in known_stripped_commands


def _contains_question(text: str) -> bool:
    lowered = text.lower()
    markers = ("?", "？", "吗", "么", "呢", "如何", "怎么", "为什么", "咋", "啥")
    return any(marker in lowered for marker in markers)


def _contains_help_intent(text: str) -> bool:
    markers = ("帮我", "求助", "看看", "解释", "分析", "怎么办", "咋办", "帮忙")
    return any(marker in text for marker in markers)


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)
