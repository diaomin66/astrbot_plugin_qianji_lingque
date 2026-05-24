from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator

from .context import ContextStore
from .decision import DecisionEngine
from .config import PluginConfig, mode_label, parse_mode
from .event_utils import group_id_from_event, snapshot_from_event
from .llm import LLMClient, extract_chain_text


PENDING_EXTRA_KEY = "qianji_lingque_pending"
EXTERNAL_LLM_EXTRA_KEY = "qianji_lingque_external_llm_recorded"
SUPPORTED_PLATFORM_IDS = {"aiocqhttp"}


@dataclass(frozen=True)
class PendingReply:
    group_id: str
    token: str
    confidence: float
    reason: str
    is_direct: bool
    started_at: float
    request_id: int
    status: str
    response_text: str = ""
    previous_streaming: Any = None
    had_send_before: bool = False


class QianjiLingqueRuntime:
    def __init__(self, context: Any, config: PluginConfig) -> None:
        self.context = context
        self.config = config
        self.context_store = ContextStore(config.max_context_messages)
        self.decision_engine = DecisionEngine()
        self.llm_client = LLMClient(context)
        self._pending_replies: dict[str, PendingReply] = {}
        self._pending_events: dict[str, Any] = {}
        self._timer_handles: set[Any] = set()
        self._timer_handles_by_token: dict[str, set[Any]] = {}

    async def handle_group_message(self, event: Any) -> AsyncIterator[Any]:
        if not _is_supported_platform(event):
            if False:
                yield None
            return
        snapshot = snapshot_from_event(event, self.config.bot_aliases)
        if _looks_like_chat_command(snapshot.text):
            if False:
                yield None
            return
        group_key = _state_group_id(event, snapshot.group_id)
        if not snapshot.group_id or not _is_group_enabled(self.config, snapshot.group_id, group_key):
            if False:
                yield None
            return
        state = self.context_store.get_group(group_key)
        decision_snapshot = replace(snapshot, group_id=group_key)
        decision_config = _decision_config_for_group(self.config, snapshot.group_id, group_key)
        decision = self.decision_engine.decide(decision_snapshot, state, decision_config)
        state.last_decision = f"{decision.action} ({decision.confidence:.2f})：{decision.reason}"
        final_action = decision.action
        final_reason = decision.reason

        if decision.is_gray_area:
            if _wants_bot(event, snapshot):
                final_action = "reply"
                final_reason = "灰区消息被点名或唤醒，交给 AstrBot 会话模型。"
            else:
                final_action = "wait"
                final_reason = "灰区消息，为避免阻塞插件链，本轮不调用模型。"
            state.last_decision = f"{final_action} ({decision.confidence:.2f})：{final_reason}"
        elif final_action == "wait" and _wants_bot(event, snapshot):
            final_action = "reply"
            final_reason = "消息被点名或唤醒，交给 AstrBot 会话模型。"
            state.last_decision = f"{final_action} ({decision.confidence:.2f})：{final_reason}"

        if (
            final_action == "reply"
            and not self.config.takeover_explicit_mentions
            and (snapshot.is_explicit_to_bot or getattr(event, "is_at_or_wake_command", False))
        ):
            state.last_decision = "ignore (0.00)：明确点名接管已关闭，交给 AstrBot 默认链路。"
            if False:
                yield None
            return

        if final_action != "reply":
            if _should_record_context(decision):
                state.append_user_message(decision_snapshot)
            if False:
                yield None
            return

        wants_bot = _wants_bot(event, snapshot)
        take_over_event = self._should_take_over_event(event, snapshot)
        pending = self._blocking_pending(group_key, wants_bot, take_over_event)
        if pending:
            state.last_decision = f"wait ({decision.confidence:.2f})：同群已有回复生成中。"
            if _should_record_context(decision):
                state.append_user_message(decision_snapshot)
            suppress_default = take_over_event or bool(getattr(event, "is_at_or_wake_command", False))
            if wants_bot or suppress_default:
                if suppress_default:
                    _suppress_default_llm(event)
                yield event.plain_result("上一条还在生成，我先不叠加请求。")
                if take_over_event:
                    event.stop_event()
            if False:
                yield None
            return

        token = uuid.uuid4().hex
        previous_streaming = _get_raw_extra(event, "enable_streaming")
        had_send_before = _event_has_send_operation(event)
        self._pending_replies[token] = PendingReply(
            group_id=group_key,
            token=token,
            confidence=decision.confidence,
            reason=final_reason,
            is_direct=wants_bot,
            started_at=time.time(),
            request_id=0,
            status="starting",
            previous_streaming=previous_streaming,
            had_send_before=had_send_before,
        )
        self._schedule_pending_expiry(token)
        reply_request = await self.llm_client.build_reply_request(event, snapshot, state, final_reason)
        if reply_request is None:
            self._pending_replies.pop(token, None)
            self._cancel_timers_for_token(token)
            _clear_qianji_extra(event)
            detail = self.llm_client.last_error or "无法取得 AstrBot 会话。"
            state.last_decision = f"ignore (0.00)：{detail} 已放弃本轮回复。"
            if False:
                yield None
            return
        if token not in self._pending_replies:
            self._cancel_timers_for_token(token)
            _clear_qianji_extra(event)
            state.last_decision = "ignore (0.00)：回复构建已被更新的直接请求抢占。"
            if False:
                yield None
            return

        request_id = id(reply_request)
        _set_pending_payload(
            event,
            {
                "group_id": group_key,
                "token": token,
                "request_id": request_id,
                "confidence": decision.confidence,
                "reason": final_reason,
                "is_direct": wants_bot,
                "had_send_before": had_send_before,
            },
        )
        _set_extra(event, "enable_streaming", False)
        self._pending_replies[token] = PendingReply(
            group_id=group_key,
            token=token,
            confidence=decision.confidence,
            reason=final_reason,
            is_direct=wants_bot,
            started_at=time.time(),
            request_id=request_id,
            status="starting",
            previous_streaming=previous_streaming,
            had_send_before=had_send_before,
        )
        self._pending_events[token] = event
        state.append_user_message(decision_snapshot)
        self._schedule_pending_expiry(token, replace=True)
        suppress_default = take_over_event or bool(getattr(event, "is_at_or_wake_command", False))
        if suppress_default:
            _suppress_default_llm(event)
        state.last_decision = f"reply ({decision.confidence:.2f})：{final_reason}"
        try:
            yield reply_request
        finally:
            _restore_streaming_extra(event, previous_streaming)
            pending_after_agent = self._pending_replies.get(token)
            if pending_after_agent is not None and pending_after_agent.status == "starting":
                self._pending_replies.pop(token, None)
                self._cancel_timers_for_token(token)
                self._cleanup_pending_event(token, pending_after_agent)
                _clear_qianji_extra(event)
                state.last_decision = "ignore (0.00)：AstrBot 未确认 LLM 请求，已清理 pending。"
            elif pending_after_agent is None:
                self._cancel_timers_for_token(token)
                self._pending_events.pop(token, None)
                _clear_qianji_extra(event)
            if take_over_event:
                event.stop_event()

    def record_llm_request(self, event: Any, request: Any) -> None:
        payload = _get_pending_payload(event)
        group_id = str(payload.get("group_id", "") or "")
        token = str(payload.get("token", "") or "")
        if not group_id or not token:
            self.record_external_llm_request(event)
            return
        request_id = _payload_int(payload, "request_id")
        if request_id and id(request) != request_id:
            return
        pending = self._pending_replies.get(token)
        confidence = _payload_float(payload, "confidence")
        reason = str(payload.get("reason", "") or "") or "已请求 AstrBot 会话模型回复。"
        is_direct = bool(payload.get("is_direct", False))
        if pending is not None and (
            pending.group_id != group_id or (pending.request_id not in {0, request_id})
        ):
            return
        self._pending_replies[token] = PendingReply(
            group_id=group_id,
            token=token,
            confidence=confidence,
            reason=reason,
            is_direct=is_direct,
            started_at=time.time(),
            request_id=request_id,
            status="active",
            previous_streaming=pending.previous_streaming if pending else None,
            had_send_before=pending.had_send_before if pending else bool(payload.get("had_send_before", False)),
        )

    def record_llm_response(self, event: Any, response: Any) -> None:
        payload = _get_pending_payload(event)
        group_id = str(payload.get("group_id", "") or "")
        token = str(payload.get("token", "") or "")
        if not group_id or not token:
            self.record_external_llm_response(event, response)
            return
        current_request_id = _provider_request_id(event)
        pending = self._pending_replies.get(token)
        if pending is None or pending.group_id != group_id:
            return
        if pending.status not in {"active", "timed_out"}:
            return
        if pending.request_id and current_request_id and pending.request_id != current_request_id:
            return

        if str(getattr(response, "role", "") or "").lower() == "err":
            state = self.context_store.peek_group(group_id)
            if state is not None:
                state.last_decision = "ignore (0.00)：AstrBot 会话模型返回错误，等待发送阶段收尾。"
            self._pending_replies[token] = PendingReply(
                group_id=pending.group_id,
                token=pending.token,
                confidence=pending.confidence,
                reason=pending.reason,
                is_direct=pending.is_direct,
                started_at=pending.started_at,
                request_id=pending.request_id,
                status="error_response",
                previous_streaming=pending.previous_streaming,
                had_send_before=pending.had_send_before,
            )
            self._schedule_send_confirmation_expiry(token)
            return

        reply_text = self.llm_client.extract_visible_reply(response)
        if not reply_text:
            state = self.context_store.peek_group(group_id)
            if state is not None:
                state.last_decision = "ignore (0.00)：AstrBot 会话模型未生成有效文本回复。"
            self._pending_replies[token] = PendingReply(
                group_id=pending.group_id,
                token=pending.token,
                confidence=pending.confidence,
                reason=pending.reason,
                is_direct=pending.is_direct,
                started_at=pending.started_at,
                request_id=pending.request_id,
                status="empty_response",
                previous_streaming=pending.previous_streaming,
                had_send_before=pending.had_send_before,
            )
            self._schedule_send_confirmation_expiry(token)
            return
        self._pending_replies[token] = PendingReply(
            group_id=pending.group_id,
            token=pending.token,
            confidence=pending.confidence,
            reason=pending.reason,
            is_direct=pending.is_direct,
            started_at=pending.started_at,
            request_id=pending.request_id,
            status="responded",
            response_text=reply_text,
            previous_streaming=pending.previous_streaming,
            had_send_before=pending.had_send_before,
        )
        self._schedule_send_confirmation_expiry(token)

    def record_after_message_sent(self, event: Any) -> None:
        payload = _get_pending_payload(event)
        group_id = str(payload.get("group_id", "") or "")
        token = str(payload.get("token", "") or "")
        if not group_id or not token:
            if _should_ignore_external_event(event):
                _delete_extra(event, EXTERNAL_LLM_EXTRA_KEY)
                return
            if _get_raw_extra(event, EXTERNAL_LLM_EXTRA_KEY):
                _delete_extra(event, EXTERNAL_LLM_EXTRA_KEY)
                return
            self.record_external_send(event)
            return
        pending = self._pending_replies.pop(token, None)
        self._cancel_timers_for_token(token)
        if pending is not None:
            _restore_streaming_extra(event, pending.previous_streaming)
        self._pending_events.pop(token, None)
        _clear_qianji_extra(event)
        if pending is None or pending.group_id != group_id:
            return
        if pending.status not in {"active", "responded", "empty_response", "error_response", "timed_out"}:
            return

        state = self.context_store.peek_group(group_id)
        if state is None:
            return
        sent_text = _extract_sent_text(event)
        send_happened_after_request = (
            _event_has_send_operation(event) and not pending.had_send_before
        )
        if not send_happened_after_request:
            if pending.status == "active":
                if not pending.had_send_before:
                    state.mark_bot_attempt(time.time())
                    state.last_decision = "ignore (0.00)：AstrBot 未返回正常 LLM 响应，已进入冷却但不写入群聊上下文。"
                else:
                    state.last_decision = "ignore (0.00)：发送标记早于本插件请求，未写入群聊上下文。"
            elif pending.status == "responded":
                if not pending.had_send_before:
                    state.mark_bot_attempt(time.time())
                    state.last_decision = "wait (0.00)：模型已产出但发送阶段未确认发出消息，已进入冷却。"
                else:
                    state.last_decision = "ignore (0.00)：发送标记早于本插件请求，未写入群聊上下文。"
            elif pending.status == "empty_response":
                if not pending.had_send_before:
                    state.mark_bot_attempt(time.time())
                    state.last_decision = "ignore (0.00)：AstrBot 会话模型未生成有效文本回复，已进入冷却。"
                else:
                    state.last_decision = "ignore (0.00)：发送标记早于本插件请求，未写入群聊上下文。"
            elif pending.status == "error_response":
                state.mark_bot_attempt(time.time())
                state.last_decision = "ignore (0.00)：AstrBot 会话模型返回错误，已进入冷却但不写入群聊上下文。"
            else:
                state.last_decision = "ignore (0.00)：发送阶段未确认发出消息，未写入群聊上下文。"
            return
        if (
            pending.status in {"active", "error_response"}
            or pending.status == "timed_out"
            or _looks_like_llm_error_text(sent_text)
        ):
            state.mark_bot_attempt(time.time())
            state.last_decision = "ignore (0.00)：AstrBot 会话模型未返回正常响应，已进入冷却但不写入群聊上下文。"
            return
        if not sent_text:
            sent_text = pending.response_text
        if not sent_text:
            state.mark_bot_attempt(time.time())
            state.last_decision = "ignore (0.00)：发送后未发现可记录的文本回复，已进入冷却。"
            return
        state.append_bot_reply(sent_text, time.time())
        state.last_decision = f"reply ({pending.confidence:.2f})：{pending.reason}"

    def _should_take_over_event(self, event: Any, snapshot: Any) -> bool:
        del event
        return bool(self.config.takeover_explicit_mentions and snapshot.is_explicit_to_bot)

    def _blocking_pending(
        self,
        group_id: str,
        incoming_wants_bot: bool,
        incoming_takeover: bool,
    ) -> PendingReply | None:
        self._purge_expired_pending()
        active = [
            (token, pending)
            for token, pending in self._pending_replies.items()
            if pending.group_id == group_id and pending.status != "empty_response"
        ]
        if not active:
            return None
        if incoming_takeover or incoming_wants_bot:
            for token, pending in list(active):
                if pending.status == "starting" and not pending.is_direct:
                    self._pending_replies.pop(token, None)
                    self._cancel_timers_for_token(token)
                    self._cleanup_pending_event(token, pending)
                    active.remove((token, pending))
            if not active:
                return None
        return active[0][1]

    def _purge_expired_pending(self) -> None:
        now = time.time()
        expired_tokens: list[str] = []
        for token, pending in self._pending_replies.items():
            age = now - pending.started_at
            if pending.status == "timed_out" and age > self.config.reply_timeout_seconds * 3:
                expired_tokens.append(token)
        for token in expired_tokens:
            pending = self._pending_replies.pop(token, None)
            if pending is not None:
                self._cancel_timers_for_token(token)
                self._cleanup_pending_event(token, pending)

    def _schedule_pending_expiry(self, token: str, *, replace: bool = False) -> None:
        self._schedule_timer(
            self.config.reply_timeout_seconds,
            self._expire_pending_token,
            token,
            replace=replace,
        )

    def _schedule_send_confirmation_expiry(self, token: str) -> None:
        delay = max(10.0, self.config.reply_timeout_seconds * 3)
        self._schedule_timer(
            delay,
            self._expire_unconfirmed_response_token,
            token,
            replace=True,
        )

    def _schedule_timer(self, delay: float, callback: Any, token: str, *, replace: bool = False) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if replace:
            self._cancel_timers_for_token(token)
        handle: Any = None

        def run_callback() -> None:
            self._timer_handles.discard(handle)
            token_handles = self._timer_handles_by_token.get(token)
            if token_handles is not None:
                token_handles.discard(handle)
                if not token_handles:
                    self._timer_handles_by_token.pop(token, None)
            callback(token)

        handle = loop.call_later(delay, run_callback)
        self._timer_handles.add(handle)
        self._timer_handles_by_token.setdefault(token, set()).add(handle)

    def _cancel_timers_for_token(self, token: str) -> None:
        handles = self._timer_handles_by_token.pop(token, set())
        for handle in list(handles):
            self._timer_handles.discard(handle)
            cancel = getattr(handle, "cancel", None)
            if callable(cancel):
                cancel()

    def _expire_unconfirmed_response_token(self, token: str) -> None:
        pending = self._pending_replies.get(token)
        if pending is None or pending.status not in {"responded", "empty_response", "error_response"}:
            return
        self._finalize_unconfirmed_response(token, pending)

    def _expire_pending_token(self, token: str) -> None:
        pending = self._pending_replies.get(token)
        if pending is None:
            return
        age = time.time() - pending.started_at
        if pending.status == "starting" and age < self.config.reply_timeout_seconds * 3:
            self._schedule_pending_expiry(token, replace=True)
            return
        if age < self.config.reply_timeout_seconds:
            self._schedule_pending_expiry(token, replace=True)
            return
        if pending.status == "timed_out" and age < self.config.reply_timeout_seconds * 3:
            self._schedule_pending_expiry(token, replace=True)
            return
        state = self.context_store.peek_group(pending.group_id)
        if pending.status == "active":
            self._pending_replies[token] = PendingReply(
                group_id=pending.group_id,
                token=pending.token,
                confidence=pending.confidence,
                reason=pending.reason,
                is_direct=pending.is_direct,
                started_at=pending.started_at,
                request_id=pending.request_id,
                status="timed_out",
                response_text=pending.response_text,
                previous_streaming=pending.previous_streaming,
                had_send_before=pending.had_send_before,
            )
            if state is not None:
                state.last_decision = "wait (0.00)：回复状态超时，保留晚到发送记录并继续节流。"
            self._schedule_pending_expiry(token, replace=True)
            return
        if pending.status in {"responded", "empty_response", "error_response"}:
            self._finalize_unconfirmed_response(token, pending)
            return
        self._pending_replies.pop(token, None)
        self._cancel_timers_for_token(token)
        self._cleanup_pending_event(token, pending)
        if state is not None:
            state.last_decision = "ignore (0.00)：回复状态超时，已清理 pending。"

    def _finalize_unconfirmed_response(self, token: str, pending: PendingReply) -> None:
        self._pending_replies.pop(token, None)
        self._cancel_timers_for_token(token)
        self._cleanup_pending_event(token, pending)
        state = self.context_store.peek_group(pending.group_id)
        if state is None:
            return
        state.mark_bot_attempt(time.time())
        if pending.status == "responded":
            state.last_decision = "wait (0.00)：模型已产出但发送钩子未触发，已进入冷却并清理 pending。"
        elif pending.status == "error_response":
            state.last_decision = "ignore (0.00)：模型返回错误且发送钩子未触发，已进入冷却并清理 pending。"
        else:
            state.last_decision = "ignore (0.00)：模型空回复且发送钩子未触发，已进入冷却并清理 pending。"

    def _cleanup_pending_event(self, token: str, pending: PendingReply) -> None:
        event = self._pending_events.pop(token, None)
        if event is None:
            return
        _restore_streaming_extra(event, pending.previous_streaming)
        _clear_qianji_extra(event)

    def prepare_agent_run(self, event: Any, run_context: Any) -> None:
        payload = _get_pending_payload(event)
        request_id = _payload_int(payload, "request_id")
        request = _provider_request(event)
        if request is None or (request_id and id(request) != request_id):
            return
        _disable_agent_tools(request, run_context)

    def record_external_send(self, event: Any) -> None:
        if not _event_has_send_operation(event):
            return
        if _should_ignore_external_event(event):
            return
        group_id = _state_group_id(event, _get_group_id(event))
        state = self.context_store.peek_group(group_id)
        if state is None:
            return
        sent_text = _extract_sent_text(event)
        if sent_text:
            state.append_bot_context(sent_text, time.time())
        state.mark_bot_attempt(time.time())

    def record_external_llm_request(self, event: Any) -> None:
        if _should_ignore_external_event(event):
            return
        group_id = _state_group_id(event, _get_group_id(event))
        state = self.context_store.peek_group(group_id)
        if state is not None:
            state.mark_bot_attempt(time.time())

    def record_external_llm_response(self, event: Any, response: Any) -> None:
        if _should_ignore_external_event(event):
            return
        group_id = _state_group_id(event, _get_group_id(event))
        state = self.context_store.peek_group(group_id)
        if state is None:
            return
        role = str(getattr(response, "role", "") or "").lower()
        reply_text = self.llm_client.extract_visible_reply(response)
        if reply_text:
            state.append_bot_context(reply_text, time.time())
        if reply_text or role == "err":
            state.mark_bot_attempt(time.time())
            _set_extra(event, EXTERNAL_LLM_EXTRA_KEY, True)

    def terminate(self) -> None:
        for handle in list(self._timer_handles):
            cancel = getattr(handle, "cancel", None)
            if callable(cancel):
                cancel()
        self._timer_handles.clear()
        self._timer_handles_by_token.clear()
        for token, pending in list(self._pending_replies.items()):
            self._cleanup_pending_event(token, pending)
        self._pending_replies.clear()
        self._pending_events.clear()

    def render_status(self, event: Any) -> str:
        group_id = _get_group_id(event)
        enabled = self.config.is_group_enabled(group_id)
        state_group_id = _state_group_id(event, group_id)
        enabled = _is_group_enabled(self.config, group_id, state_group_id)
        effective_mode = _effective_mode(self.config, group_id, state_group_id)
        return (
            "千机聆阙状态\n"
            f"当前群：{group_id or '未知'}\n"
            f"总开关：{'开启' if self.config.enabled else '关闭'}\n"
            f"群开关：{'开启' if enabled else '关闭'}\n"
            f"模式：{mode_label(effective_mode)}"
        )

    def enable_group(self, event: Any) -> str:
        group_id = _get_group_id(event)
        config_group_id = _state_group_id(event, group_id)
        if not group_id:
            return "千机聆阙：请在群聊中使用这个指令。"
        if not self.config.enabled:
            return "千机聆阙：总开关已关闭，请先在 WebUI 启用插件。"
        for candidate in {group_id, config_group_id}:
            if candidate in self.config.disabled_groups:
                self.config.disabled_groups.remove(candidate)
        if self.config.enabled_groups and config_group_id not in self.config.enabled_groups:
            self.config.enabled_groups.append(config_group_id)
        self.config.save()
        return "千机聆阙：已开启当前群。"

    def disable_group(self, event: Any) -> str:
        group_id = _get_group_id(event)
        config_group_id = _state_group_id(event, group_id)
        if not group_id:
            return "千机聆阙：请在群聊中使用这个指令。"
        if config_group_id not in self.config.disabled_groups:
            self.config.disabled_groups.append(config_group_id)
        self.config.save()
        return "千机聆阙：已关闭当前群。"

    def set_mode(self, event: Any, mode: str) -> str:
        group_id = _get_group_id(event)
        config_group_id = _state_group_id(event, group_id)
        if not group_id:
            return "千机聆阙：请在群聊中使用这个指令。"
        normalized = parse_mode(mode)
        if normalized is None:
            current = mode_label(_effective_mode(self.config, group_id, config_group_id))
            return f"千机聆阙：当前是{current}模式，可切换为 安静、普通、积极。"
        self.config.group_modes[config_group_id] = normalized
        self.config.save()
        return f"千机聆阙：已切换为{mode_label(normalized)}模式。"

    def render_last_reason(self, event: Any) -> str:
        if not self.config.debug_explain_enabled:
            return "千机聆阙：原因查看已关闭。"
        group_id = _get_group_id(event)
        if not group_id:
            return "千机聆阙：请在群聊中使用这个指令。"
        state = self.context_store.peek_group(_state_group_id(event, group_id))
        if state is None:
            return "千机聆阙：还没有判定记录。"
        return f"千机聆阙：{state.last_decision}"


def _get_group_id(event: Any) -> str:
    return group_id_from_event(event)


def _state_group_id(event: Any, group_id: str) -> str:
    if not group_id:
        return ""
    platform_id = _event_platform_id(event)
    platform_name = _event_platform_name(event)
    if platform_id and platform_id != platform_name:
        return f"{platform_id}:{group_id}"
    return group_id


def _is_group_enabled(config: PluginConfig, group_id: str, state_group_id: str) -> bool:
    if not config.enabled:
        return False
    if state_group_id in config.disabled_groups or group_id in config.disabled_groups:
        return False
    if not config.enabled_groups:
        return True
    return state_group_id in config.enabled_groups or group_id in config.enabled_groups


def _effective_mode(config: PluginConfig, group_id: str, state_group_id: str) -> str:
    return config.group_modes.get(state_group_id, config.effective_mode(group_id))


def _decision_config_for_group(config: PluginConfig, group_id: str, state_group_id: str) -> PluginConfig:
    if group_id == state_group_id:
        return config
    return replace(
        config,
        enabled_groups=_mirror_group_entries(config.enabled_groups, group_id, state_group_id),
        disabled_groups=_mirror_group_entries(config.disabled_groups, group_id, state_group_id),
        group_modes=_mirror_group_mode_entries(config.group_modes, group_id, state_group_id),
        source=None,
    )


def _mirror_group_entries(values: list[str], group_id: str, state_group_id: str) -> list[str]:
    mirrored = list(values)
    if group_id in values and state_group_id not in mirrored:
        mirrored.append(state_group_id)
    if state_group_id in values and group_id not in mirrored:
        mirrored.append(group_id)
    return mirrored


def _mirror_group_mode_entries(values: dict[str, str], group_id: str, state_group_id: str) -> dict[str, str]:
    mirrored = dict(values)
    if group_id in values and state_group_id not in mirrored:
        mirrored[state_group_id] = values[group_id]
    if state_group_id in values and group_id not in mirrored:
        mirrored[group_id] = values[state_group_id]
    return mirrored


def _message_text_from_event(event: Any) -> str:
    getter = getattr(event, "get_message_str", None)
    if callable(getter):
        try:
            return str(getter() or "")
        except Exception:
            return ""
    return str(getattr(event, "message_str", "") or "")


def _should_ignore_external_event(event: Any) -> bool:
    return _looks_like_chat_command(_message_text_from_event(event))


def _looks_like_chat_command(text: str) -> bool:
    stripped = text.strip()
    normalized = " ".join(stripped.split())
    return (
        normalized.startswith(("/", "!", "！", "#", "＃", "."))
        or _is_plugin_command(normalized)
        or _looks_like_registered_command(normalized)
        or _looks_like_common_stripped_command(normalized)
    )


def _looks_like_registered_command(text: str) -> bool:
    try:
        from astrbot.core.star.star_handler import star_handlers_registry
    except Exception:
        return False
    handlers = getattr(star_handlers_registry, "_handlers", [])
    for handler in handlers:
        for event_filter in getattr(handler, "event_filters", []) or []:
            get_names = getattr(event_filter, "get_complete_command_names", None)
            if not callable(get_names):
                continue
            try:
                command_names = get_names()
            except Exception:
                continue
            for command_name in command_names:
                command = str(command_name or "").strip()
                if command and _matches_command_prefix(text, command):
                    return True
    return False


def _looks_like_common_stripped_command(text: str) -> bool:
    if not text:
        return False
    command = text.split(" ", 1)[0].strip().lower()
    known_commands = {
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
    }
    return command in known_commands


def _looks_like_llm_error_text(text: str) -> bool:
    lowered = text.strip().lower()
    markers = (
        "llm 响应错误",
        "error occurred during ai execution",
        "error type:",
        "error message:",
        "provider exploded",
    )
    return any(marker in lowered for marker in markers)


def _event_platform_name(event: Any) -> str:
    name_getter = getattr(event, "get_platform_name", None)
    if callable(name_getter):
        try:
            return str(name_getter() or "").strip()
        except Exception:
            return ""
    meta = getattr(event, "platform_meta", None)
    return str(getattr(meta, "name", "") or "").strip()


def _event_platform_id(event: Any) -> str:
    id_getter = getattr(event, "get_platform_id", None)
    if callable(id_getter):
        try:
            return str(id_getter() or "").strip()
        except Exception:
            return ""
    return str(getattr(event, "platform_id", "") or "").strip()


def _is_supported_platform(event: Any) -> bool:
    platform_name = _event_platform_name(event)
    if platform_name:
        return platform_name in SUPPORTED_PLATFORM_IDS
    meta = getattr(event, "platform_meta", None)
    meta_name = str(getattr(meta, "name", "") or "").strip()
    if meta_name:
        return meta_name in SUPPORTED_PLATFORM_IDS
    platform_id = _event_platform_id(event)
    if not platform_id:
        return True
    return not platform_id or platform_id in SUPPORTED_PLATFORM_IDS


def _is_plugin_command(text: str) -> bool:
    normalized = text.strip()
    return any(_matches_command_prefix(normalized, command) for command in ("/读空气", "/空气", "读空气", "空气"))


def _matches_command_prefix(text: str, command: str) -> bool:
    if text == command:
        return True
    return text.startswith(f"{command} ")


def _should_record_context(decision: Any) -> bool:
    if decision.action != "ignore":
        return True
    blocked_reasons = ("空消息", "指令", "忽略 bot 自己", "插件未在当前群启用")
    return not any(reason in decision.reason for reason in blocked_reasons)


def _wants_bot(event: Any, snapshot: Any) -> bool:
    return bool(snapshot.is_direct_to_bot or getattr(event, "is_at_or_wake_command", False))


def _suppress_default_llm(event: Any) -> None:
    should_call_llm = getattr(event, "should_call_llm", None)
    if callable(should_call_llm):
        # AstrBot API name is historical: True means suppress the default LLM chain.
        should_call_llm(True)


def _set_extra(event: Any, key: str, value: Any) -> None:
    setter = getattr(event, "set_extra", None)
    if callable(setter):
        setter(key, value)
        return
    extra = getattr(event, "extra", None)
    if isinstance(extra, dict):
        extra[key] = value
        return
    setattr(event, key, value)


def _get_extra(event: Any, key: str) -> str:
    value = _get_raw_extra(event, key)
    return str(value or "")


def _get_raw_extra(event: Any, key: str) -> Any:
    getter = getattr(event, "get_extra", None)
    if callable(getter):
        return getter(key)
    extra = getattr(event, "extra", None)
    if isinstance(extra, dict):
        return extra.get(key)
    return getattr(event, key, None)


def _set_pending_payload(event: Any, payload: dict[str, Any]) -> None:
    _set_extra(event, PENDING_EXTRA_KEY, payload)


def _get_pending_payload(event: Any) -> dict[str, Any]:
    payload = _get_raw_extra(event, PENDING_EXTRA_KEY)
    if isinstance(payload, dict):
        return payload
    return {}


def _provider_request_id(event: Any) -> int:
    request = _provider_request(event)
    return id(request) if request is not None else 0


def _provider_request(event: Any) -> Any:
    request = None
    getter = getattr(event, "get_extra", None)
    if callable(getter):
        request = getter("provider_request")
    if request is None:
        extra = getattr(event, "extra", None)
        if isinstance(extra, dict):
            request = extra.get("provider_request")
    return request


def _clear_qianji_extra(event: Any) -> None:
    _delete_extra(event, PENDING_EXTRA_KEY)
    keys = (
        "qianji_lingque_pending_group",
        "qianji_lingque_pending_token",
        "qianji_lingque_pending_request_id",
        "qianji_lingque_pending_confidence",
        "qianji_lingque_pending_reason",
        "qianji_lingque_pending_is_direct",
    )
    for key in keys:
        _delete_extra(event, key)


def _restore_streaming_extra(event: Any, previous_value: Any) -> None:
    if previous_value is None:
        _delete_extra(event, "enable_streaming")
        return
    _set_extra(event, "enable_streaming", previous_value)


def _delete_extra(event: Any, key: str) -> None:
    saw_extra_dict = False
    for attr_name in ("_extras", "extra"):
        extra = getattr(event, attr_name, None)
        if isinstance(extra, dict):
            saw_extra_dict = True
            extra.pop(key, None)
    if hasattr(event, key):
        try:
            delattr(event, key)
        except AttributeError:
            setattr(event, key, None)
        return
    if saw_extra_dict:
        return
    setter = getattr(event, "set_extra", None)
    if callable(setter):
        setter(key, None)


def _float_extra(event: Any, key: str) -> float:
    try:
        return float(_get_extra(event, key))
    except ValueError:
        return 0.0


def _int_extra(event: Any, key: str) -> int:
    try:
        return int(_get_extra(event, key))
    except ValueError:
        return 0


def _payload_float(payload: dict[str, Any], key: str) -> float:
    try:
        return float(payload.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _payload_int(payload: dict[str, Any], key: str) -> int:
    try:
        return int(payload.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _disable_agent_tools(request: Any, run_context: Any) -> None:
    try:
        setattr(request, "func_tool", None)
    except Exception:
        pass
    system_prompt = getattr(request, "system_prompt", None)
    if isinstance(system_prompt, str):
        try:
            setattr(request, "system_prompt", _strip_tool_prompt_text(system_prompt))
        except Exception:
            pass
    messages = getattr(run_context, "messages", None)
    if not isinstance(messages, list):
        return
    for message in messages:
        if getattr(message, "role", "") != "system":
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str):
            try:
                message.content = _strip_tool_prompt_text(content)
            except Exception:
                pass


def _strip_tool_prompt_text(text: str) -> str:
    lines = []
    skip_next_workspace_line = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("When using tools:"):
            continue
        if stripped.startswith("You MUST NOT return an empty response"):
            continue
        if stripped.startswith("You have access to the host local environment"):
            continue
        if stripped.startswith("You have access to a sandboxed environment"):
            continue
        if stripped.startswith("Current workspace you can use:"):
            skip_next_workspace_line = True
            continue
        if skip_next_workspace_line and (
            "perform all file-related operations" in stripped
            or "Unless the user explicitly" in stripped
        ):
            continue
        skip_next_workspace_line = False
        lines.append(line)
    return "\n".join(lines).strip()


def _get_event_result(event: Any) -> Any:
    result_getter = getattr(event, "get_result", None)
    return result_getter() if callable(result_getter) else getattr(event, "_result", None)


def _extract_sent_text(event: Any) -> str:
    return extract_chain_text(_get_event_result(event))


def _event_has_send_operation(event: Any) -> bool:
    return bool(getattr(event, "_has_send_oper", False))
