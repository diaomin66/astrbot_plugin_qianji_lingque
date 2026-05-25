from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
import re


@dataclass(frozen=True)
class MessageSnapshot:
    group_id: str
    sender_id: str
    sender_name: str
    message_id: str
    self_id: str
    text: str
    outline: str
    timestamp: float
    mentions_bot: bool
    replies_to_bot: bool
    ats_bot: bool = False
    uses_bot_alias: bool = False

    @property
    def is_direct_to_bot(self) -> bool:
        return self.mentions_bot or self.replies_to_bot

    @property
    def is_explicit_to_bot(self) -> bool:
        return self.ats_bot or self.replies_to_bot


def snapshot_from_event(event: Any, bot_aliases: list[str]) -> MessageSnapshot:
    text = _call_or_attr(event, "get_message_str", "message_str")
    outline = _call_or_attr(event, "get_message_outline", default=text)
    self_id = _call_or_attr(event, "get_self_id", "self_id")
    group_id = group_id_from_event(event)
    sender_id = _call_or_attr(event, "get_sender_id", "sender_id")
    sender_name = _call_or_attr(event, "get_sender_name", "sender_name") or sender_id
    message_id = _call_or_attr(event, "get_message_id", "message_id")
    timestamp = _event_timestamp(event)
    messages = _event_messages(event)
    ats_bot = _ats_bot(messages, self_id)
    uses_bot_alias = any(_contains_alias(text, alias) for alias in bot_aliases if alias)
    return MessageSnapshot(
        group_id=group_id,
        sender_id=sender_id,
        sender_name=sender_name,
        message_id=message_id,
        self_id=self_id,
        text=text.strip(),
        outline=outline.strip(),
        timestamp=timestamp,
        mentions_bot=ats_bot or uses_bot_alias,
        replies_to_bot=_replies_to_bot(messages, self_id),
        ats_bot=ats_bot,
        uses_bot_alias=uses_bot_alias,
    )


def group_id_from_event(event: Any) -> str:
    return _call_or_attr(event, "get_group_id", "group_id")


def _call_or_attr(event: Any, method_name: str, attr_name: str | None = None, default: str = "") -> str:
    method = getattr(event, method_name, None)
    if callable(method):
        try:
            result = str(method() or "")
            if result:
                return result
        except TypeError:
            pass
    if attr_name:
        direct = getattr(event, attr_name, None)
        if direct is not None:
            result = str(direct or "")
            if result:
                return result
    message_obj = getattr(event, "message_obj", None)
    if attr_name and message_obj is not None:
        nested = getattr(message_obj, attr_name, None)
        if nested is not None:
            result = str(nested or "")
            if result:
                return result
        sender = getattr(message_obj, "sender", None)
        if sender is not None:
            sender_attr = _sender_attr_for(attr_name)
            if sender_attr:
                nested_sender = getattr(sender, sender_attr, None)
                if nested_sender is not None:
                    result = str(nested_sender or "")
                    if result:
                        return result
    return default


def _event_messages(event: Any) -> list[Any]:
    getter = getattr(event, "get_messages", None)
    if callable(getter):
        try:
            messages = getter()
            if isinstance(messages, list):
                return messages
        except TypeError:
            pass
    message_obj = getattr(event, "message_obj", None)
    messages = getattr(message_obj, "message", None)
    return messages if isinstance(messages, list) else []


def _event_timestamp(event: Any) -> float:
    message_obj = getattr(event, "message_obj", None)
    raw_timestamp = getattr(message_obj, "timestamp", None)
    try:
        return float(raw_timestamp)
    except (TypeError, ValueError):
        return time.time()


def _ats_bot(messages: list[Any], self_id: str) -> bool:
    if self_id:
        for component in messages:
            name = component.__class__.__name__.lower()
            if "at" not in name:
                continue
            candidate_ids = [
                getattr(component, "qq", None),
                getattr(component, "user_id", None),
                getattr(component, "target", None),
            ]
            if any(str(candidate) == self_id for candidate in candidate_ids if candidate is not None):
                return True
    return False


def _contains_alias(text: str, alias: str) -> bool:
    if not alias:
        return False
    if _is_ascii_word(alias):
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", text, re.IGNORECASE) is not None
    pattern = re.compile(re.escape(alias))
    for match in pattern.finditer(text):
        previous_char = text[match.start() - 1] if match.start() > 0 else ""
        next_char = text[match.end()] if match.end() < len(text) else ""
        if _is_wake_boundary(previous_char, before=True) and _is_wake_boundary(next_char, before=False):
            return True
        if _is_wake_boundary(previous_char, before=True) and _is_common_chinese_wake_tail(next_char):
            return True
    return False


def _is_ascii_word(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_]+", value))


def _is_wake_boundary(char: str, *, before: bool) -> bool:
    if not char:
        return True
    if char.isspace():
        return True
    if char in "@,，.。!！?？:：;；、~～-—()（）[]【】<>《》":
        return True
    return False


def _is_common_chinese_wake_tail(char: str) -> bool:
    return char in {"你", "帮", "看", "来", "请", "给"}


def _sender_attr_for(attr_name: str) -> str | None:
    if attr_name == "sender_id":
        return "user_id"
    if attr_name == "sender_name":
        return "nickname"
    return None


def _replies_to_bot(messages: list[Any], self_id: str) -> bool:
    if not self_id:
        return False
    for component in messages:
        name = component.__class__.__name__.lower()
        if "reply" not in name and "quote" not in name:
            continue
        candidate_ids = [
            getattr(component, "sender_id", None),
            getattr(component, "user_id", None),
            getattr(component, "qq", None),
        ]
        if any(str(candidate) == self_id for candidate in candidate_ids if candidate is not None):
            return True
    return False
