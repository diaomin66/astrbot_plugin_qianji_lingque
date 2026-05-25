from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Deque

from .event_utils import MessageSnapshot


@dataclass
class ChatMessage:
    sender_id: str
    sender_name: str
    text: str
    timestamp: float
    is_bot: bool = False


@dataclass
class GroupState:
    group_id: str
    max_messages: int
    messages: Deque[ChatMessage] = field(default_factory=deque)
    last_bot_reply_at: float = 0.0
    last_decision: str = "还没有判定记录。"
    updated_at: float = field(default_factory=time.time)

    def append_user_message(self, snapshot: MessageSnapshot) -> None:
        self.touch()
        self._append(
            ChatMessage(
                sender_id=snapshot.sender_id,
                sender_name=snapshot.sender_name,
                text=snapshot.text,
                timestamp=snapshot.timestamp,
            ),
        )

    def append_bot_reply(self, text: str, timestamp: float) -> None:
        self.mark_bot_reply(text, timestamp, update_cooldown=True)

    def append_bot_context(self, text: str, timestamp: float) -> None:
        self.mark_bot_reply(text, timestamp, update_cooldown=False)

    def mark_bot_attempt(self, timestamp: float) -> None:
        self.touch()
        self.last_bot_reply_at = timestamp

    def mark_bot_reply(self, text: str, timestamp: float, *, update_cooldown: bool) -> None:
        self.touch()
        if update_cooldown:
            self.last_bot_reply_at = timestamp
        self._append(
            ChatMessage(
                sender_id="__bot__",
                sender_name="bot",
                text=text,
                timestamp=timestamp,
                is_bot=True,
            ),
        )

    def last_user_message(self) -> ChatMessage | None:
        for message in reversed(self.messages):
            if not message.is_bot:
                return message
        return None

    def last_message(self) -> ChatMessage | None:
        if not self.messages:
            return None
        return self.messages[-1]

    def last_bot_message(self) -> ChatMessage | None:
        for message in reversed(self.messages):
            if message.is_bot:
                return message
        return None

    def recent_user_ids(self, limit: int = 6) -> set[str]:
        users: set[str] = set()
        for message in list(self.messages)[-limit:]:
            if not message.is_bot and message.sender_id:
                users.add(message.sender_id)
        return users

    def render_recent_context(self, limit: int = 12) -> str:
        lines: list[str] = []
        for message in list(self.messages)[-limit:]:
            name = "bot" if message.is_bot else message.sender_name or message.sender_id or "群友"
            text = message.text.replace("\n", " ").strip()
            if text:
                lines.append(f"{name}: {text}")
        return "\n".join(lines)

    def _append(self, message: ChatMessage) -> None:
        self.touch()
        self.messages.append(message)
        while len(self.messages) > self.max_messages:
            self.messages.popleft()

    def touch(self) -> None:
        self.updated_at = time.time()


class ContextStore:
    def __init__(
        self,
        max_messages: int,
        *,
        max_groups: int = 200,
        group_ttl_seconds: float = 86400.0,
    ) -> None:
        self.max_messages = max_messages
        self.max_groups = max_groups
        self.group_ttl_seconds = group_ttl_seconds
        self.protected_group_ids: set[str] = set()
        self._groups: dict[str, GroupState] = {}

    def get_group(self, group_id: str) -> GroupState:
        self._purge_stale_groups()
        state = self._groups.get(group_id)
        if state is None:
            state = GroupState(group_id=group_id, max_messages=self.max_messages)
            self._groups[group_id] = state
        state.max_messages = self.max_messages
        state.touch()
        self._enforce_group_limit(group_id)
        return state

    def peek_group(self, group_id: str) -> GroupState | None:
        self._purge_stale_groups()
        return self._groups.get(group_id)

    def configure(self, *, max_messages: int, max_groups: int, group_ttl_seconds: float) -> None:
        self.max_messages = max_messages
        self.max_groups = max_groups
        self.group_ttl_seconds = group_ttl_seconds
        self._purge_stale_groups()
        self._enforce_group_limit("")

    def set_protected_groups(self, group_ids: set[str]) -> None:
        self.protected_group_ids = set(group_ids)
        self._enforce_group_limit("")

    def _purge_stale_groups(self) -> None:
        if self.group_ttl_seconds <= 0:
            return
        cutoff = time.time() - self.group_ttl_seconds
        for group_id, state in list(self._groups.items()):
            if group_id in self.protected_group_ids:
                continue
            if state.updated_at < cutoff:
                self._groups.pop(group_id, None)

    def _enforce_group_limit(self, protected_group_id: str) -> None:
        if self.max_groups <= 0:
            self._groups.clear()
            return
        while len(self._groups) > self.max_groups:
            candidates = [
                (group_id, state)
                for group_id, state in self._groups.items()
                if group_id != protected_group_id and group_id not in self.protected_group_ids
            ]
            if not candidates:
                return
            oldest_group_id, _ = min(candidates, key=lambda item: item[1].updated_at)
            self._groups.pop(oldest_group_id, None)
