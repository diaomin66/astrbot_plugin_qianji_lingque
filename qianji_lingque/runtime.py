from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator

from .context import ContextStore
from .decision import DecisionEngine
from .config import PluginConfig, mode_label, parse_mode
from .event_utils import group_id_from_event, snapshot_from_event
from .llm import LLMClient, extract_chain_text

try:
    from astrbot.api import logger
except ImportError:
    logger = logging.getLogger("astrbot_plugin_qianji_lingque")

try:
    from astrbot.core.astr_main_agent_resources import (
        SANDBOX_MODE_PROMPT,
        TOOL_CALL_PROMPT,
        TOOL_CALL_PROMPT_SKILLS_LIKE_MODE,
    )
except Exception:
    SANDBOX_MODE_PROMPT = ""
    TOOL_CALL_PROMPT = ""
    TOOL_CALL_PROMPT_SKILLS_LIKE_MODE = ""

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_workspaces_path
except Exception:
    get_astrbot_workspaces_path = None


PENDING_EXTRA_KEY = "qianji_lingque_pending"
EXTERNAL_LLM_EXTRA_KEY = "qianji_lingque_external_llm_recorded"
EXTERNAL_LLM_REQUEST_EXTRA_KEY = "qianji_lingque_external_llm_requested"
SUPPORTED_PLATFORM_IDS = {"aiocqhttp"}
SDK_TOOL_PROMPTS = tuple(
    prompt.strip()
    for prompt in (SANDBOX_MODE_PROMPT, TOOL_CALL_PROMPT, TOOL_CALL_PROMPT_SKILLS_LIKE_MODE)
    if isinstance(prompt, str) and prompt.strip()
)
KNOWN_TOOL_PROMPTS = {
    "When using tools: never return an empty response; follow schemas.",
    "When using tools: never return an empty response; briefly explain the purpose before calling a tool; follow the tool schema exactly and do not invent parameters; after execution, briefly summarize the result for the user; keep the conversation style consistent.",
    "You MUST NOT return an empty response, especially after invoking a tool.",
    "You MUST NOT return an empty response, especially after invoking a tool. Before calling any tool, provide a brief explanatory message to the user stating the purpose of the tool call. Tool schemas are provided in two stages: first only name and description; if you decide to use a tool, the full parameter schema will be provided in a follow-up step. Do not guess arguments before you see the schema. After the tool call is completed, you must briefly summarize the results returned by the tool for the user. Keep the role-play and style consistent throughout the conversation.",
    "You have access to the host local environment and can execute shell commands.",
    "You have access to a sandboxed environment and can execute shell commands and Python code securely.",
    "User has not enabled the Computer Use feature.",
}
ASTRBOT_WORKSPACES_PATH = (
    os.path.realpath(get_astrbot_workspaces_path())
    if callable(get_astrbot_workspaces_path)
    else ""
)


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
        self.context_store = ContextStore(
            config.max_context_messages,
            max_groups=config.max_tracked_groups,
            group_ttl_seconds=config.group_ttl_seconds,
        )
        self.decision_engine = DecisionEngine()
        self.llm_client = LLMClient(context)
        self._pending_replies: dict[str, PendingReply] = {}
        self._pending_events: dict[str, Any] = {}
        self._timer_handles: set[Any] = set()
        self._timer_handles_by_token: dict[str, set[Any]] = {}

    async def handle_group_message(self, event: Any) -> AsyncIterator[Any]:
        if not _is_supported_platform(event):
            self._log_decision(event, action="ignore", will_call_llm=False, reason="平台不支持，仅支持 aiocqhttp。")
            if False:
                yield None
            return
        self._sync_protected_context_groups()
        self.context_store.configure(
            max_messages=self.config.max_context_messages,
            max_groups=self.config.max_tracked_groups,
            group_ttl_seconds=self.config.group_ttl_seconds,
        )
        if _event_has_send_operation(event) and not _get_pending_payload(event):
            self._log_decision(event, action="ignore", will_call_llm=False, reason="事件已有发送动作，避免重复处理。")
            if False:
                yield None
            return
        snapshot = snapshot_from_event(event, self.config.bot_aliases)
        if _looks_like_chat_command(snapshot.text):
            self._log_decision(
                event,
                action="ignore",
                will_call_llm=False,
                reason="识别为指令或其他插件命令，交给命令链路。",
                group_id=snapshot.group_id,
                text=snapshot.text,
            )
            if False:
                yield None
            return
        if _get_raw_extra(event, EXTERNAL_LLM_REQUEST_EXTRA_KEY):
            self._log_decision(
                event,
                action="ignore",
                will_call_llm=False,
                reason="同一事件已有其他链路请求 LLM，本插件避让。",
                group_id=snapshot.group_id,
                text=snapshot.text,
            )
            if False:
                yield None
            return
        group_key = _state_group_id(event, snapshot.group_id)
        if not snapshot.group_id or not _is_group_enabled(self.config, snapshot.group_id, group_key):
            self._log_decision(
                event,
                action="ignore",
                will_call_llm=False,
                reason="插件未在当前群启用或缺少群号。",
                group_id=snapshot.group_id,
                state_group_id=group_key,
                text=snapshot.text,
            )
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
        final_confidence = decision.confidence
        timing_gate_called = False
        direct_reply_text = ""
        direct_reply_reason = ""
        direct_reply_called = False

        if decision.is_gray_area:
            if _wants_bot(event, snapshot):
                final_action = "reply"
                final_reason = "灰区消息被点名或唤醒，交给 AstrBot 会话模型。"
            elif not self.config.llm_gate_enabled:
                if _gray_local_fallback_should_reply(decision, snapshot, self.config):
                    final_action = "reply"
                    final_confidence = _local_fallback_confidence(decision)
                    final_reason = "灰区消息带求助信号，节奏判断关闭时本地兜底接话。"
                else:
                    final_action = "wait"
                    final_reason = "灰区消息，轻量节奏判断已关闭，本轮不调用模型。"
            elif not _gray_message_is_worth_llm_gate(decision, snapshot, self.config):
                if _gray_local_fallback_should_reply(decision, snapshot, self.config):
                    final_action = "reply"
                    final_confidence = _local_fallback_confidence(decision)
                    final_reason = "灰区消息带求助信号，本地兜底接话。"
                else:
                    final_action = "wait"
                    final_reason = "灰区消息缺少求助信号，本轮不调用模型。"
            else:
                if not await self.llm_client.reply_request_supported(event):
                    unsupported_reason = self.llm_client.last_error or "当前回复链路不可用"
                    if _gray_local_fallback_should_reply(decision, snapshot, self.config):
                        direct_reply_reason = f"灰区节奏判断未调用，主动回复链路不可用，改用直发兜底：{unsupported_reason}"
                        direct_reply_result = await self.llm_client.generate_direct_reply(
                            event,
                            decision_snapshot,
                            state,
                            decision_reason=direct_reply_reason,
                            timeout_seconds=self.config.llm_gate_timeout_seconds,
                        )
                        direct_reply_text = direct_reply_result.text
                        direct_reply_called = direct_reply_result.called_llm
                        if direct_reply_text:
                            final_action = "reply"
                            final_confidence = _local_fallback_confidence(decision)
                            final_reason = direct_reply_reason
                        else:
                            final_action = "wait"
                            final_reason = (
                                f"灰区节奏判断未调用，直发兜底失败，本轮放行默认链路但默认链路未必会回复"
                                f"（明确点名={'是' if _wants_bot(event, snapshot) else '否'}）："
                                f"{direct_reply_result.reason or unsupported_reason}"
                            )
                    else:
                        final_action = "wait"
                        final_reason = (
                            f"灰区节奏判断未调用，本轮放行默认链路但默认链路未必会回复"
                            f"（明确点名={'是' if _wants_bot(event, snapshot) else '否'}）：{unsupported_reason}"
                        )
                else:
                    timing_result = await self.llm_client.judge_timing(
                        event,
                        decision_snapshot,
                        state,
                        local_score=decision.confidence,
                        local_reason=decision.reason,
                        mode_label=mode_label(_effective_mode(self.config, snapshot.group_id, group_key)),
                        timeout_seconds=self.config.llm_gate_timeout_seconds,
                    )
                    timing_gate_called = timing_result.called_llm
                    final_action = timing_result.action
                    final_confidence = timing_result.confidence
                    final_reason = f"灰区 TimingGate：{timing_result.reason}"
                    should_fallback_after_gate = (
                        final_action != "reply"
                        and _timing_gate_should_fallback(timing_result, decision, snapshot, self.config)
                        and _gray_local_fallback_should_reply(decision, snapshot, self.config)
                    )
                    if should_fallback_after_gate:
                        final_action = "reply"
                        final_confidence = _local_fallback_confidence(decision)
                        final_reason = f"灰区 TimingGate 后本地兜底接话：{timing_result.reason}"
                    self._log_lifecycle(
                        "TimingGate节奏判断",
                        event,
                        group_id=group_key,
                        detail=(
                            f"实际调用LLM={'是' if timing_gate_called else '否'}；"
                            f"TimingGate动作={timing_result.action}；最终动作={final_action}；"
                            f"TimingGate分数={timing_result.confidence:.2f}；最终分数={final_confidence:.2f}；"
                            f"原因={timing_result.reason}"
                        ),
                    )
            state.last_decision = f"{final_action} ({final_confidence:.2f})：{final_reason}"
        elif final_action == "wait" and _wants_bot(event, snapshot):
            final_action = "reply"
            final_reason = "消息被点名或唤醒，交给 AstrBot 会话模型。"
            state.last_decision = f"{final_action} ({final_confidence:.2f})：{final_reason}"

        if (
            final_action == "reply"
            and not self.config.takeover_explicit_mentions
            and (snapshot.is_explicit_to_bot or getattr(event, "is_at_or_wake_command", False))
        ):
            state.last_decision = "ignore (0.00)：明确点名接管已关闭，交给 AstrBot 默认链路。"
            self._log_decision(
                event,
                action="ignore",
                will_call_llm=False,
                reason="明确点名接管已关闭，交给 AstrBot 默认链路。",
                confidence=0.0,
                group_id=snapshot.group_id,
                state_group_id=group_key,
                text=snapshot.text,
            )
            if False:
                yield None
            return

        if final_action != "reply":
            if _should_record_context(decision):
                state.append_user_message(decision_snapshot)
            self._log_decision(
                event,
                action=final_action,
                will_call_llm=False,
                timing_gate_called=timing_gate_called,
                reason=final_reason,
                confidence=final_confidence,
                group_id=snapshot.group_id,
                state_group_id=group_key,
                text=snapshot.text,
            )
            if False:
                yield None
            return

        if direct_reply_text:
            token = uuid.uuid4().hex
            previous_streaming = _get_raw_extra(event, "enable_streaming")
            had_send_before = _event_has_send_operation(event)
            state.append_user_message(decision_snapshot)
            state.last_decision = f"reply ({final_confidence:.2f})：{final_reason}"
            _set_pending_payload(
                event,
                {
                    "group_id": group_key,
                    "token": token,
                    "request_id": 0,
                    "confidence": final_confidence,
                    "reason": final_reason,
                    "is_direct": True,
                    "had_send_before": had_send_before,
                },
            )
            self._pending_replies[token] = PendingReply(
                group_id=group_key,
                token=token,
                confidence=final_confidence,
                reason=final_reason,
                is_direct=True,
                started_at=time.time(),
                request_id=0,
                status="responded",
                response_text=direct_reply_text,
                previous_streaming=previous_streaming,
                had_send_before=had_send_before,
            )
            self._pending_events[token] = event
            self._sync_protected_context_groups()
            self._schedule_send_confirmation_expiry(token)
            self._log_decision(
                event,
                action="reply",
                will_call_llm=False,
                timing_gate_called=timing_gate_called,
                reason=f"{final_reason}；回复链路=直发；直发LLM实际={'是' if direct_reply_called else '否'}。",
                confidence=final_confidence,
                group_id=snapshot.group_id,
                state_group_id=group_key,
                text=snapshot.text,
            )
            try:
                yield event.plain_result(direct_reply_text)
            finally:
                event.stop_event()
            return

        wants_bot = _wants_bot(event, snapshot)
        take_over_event = self._should_take_over_event(event, snapshot)
        pending = self._blocking_pending(group_key, wants_bot, take_over_event)
        if pending:
            state.last_decision = f"wait ({final_confidence:.2f})：同群已有回复生成中。"
            if _should_record_context(decision):
                state.append_user_message(decision_snapshot)
            suppress_default = take_over_event or bool(getattr(event, "is_at_or_wake_command", False))
            self._log_decision(
                event,
                action="wait",
                will_call_llm=False,
                reason="同群已有回复生成中，本轮不再调用 LLM。",
                confidence=final_confidence,
                group_id=snapshot.group_id,
                state_group_id=group_key,
                text=snapshot.text,
            )
            if suppress_default:
                _suppress_default_llm(event)
                yield event.plain_result("上一条还在生成，我先不叠加请求。")
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
            confidence=final_confidence,
            reason=final_reason,
            is_direct=wants_bot,
            started_at=time.time(),
            request_id=0,
            status="starting",
            previous_streaming=previous_streaming,
            had_send_before=had_send_before,
        )
        self._sync_protected_context_groups()
        self._schedule_pending_expiry(token)
        reply_request = await self.llm_client.build_reply_request(event, snapshot, state, final_reason)
        if reply_request is None:
            self._pending_replies.pop(token, None)
            self._sync_protected_context_groups()
            self._cancel_timers_for_token(token)
            _clear_qianji_extra(event)
            detail = self.llm_client.last_error or "无法取得 AstrBot 会话。"
            state.last_decision = f"ignore (0.00)：{detail} 已放弃本轮回复。"
            self._log_decision(
                event,
                action="ignore",
                will_call_llm=False,
                reason=f"{detail} 已放弃本轮回复。",
                confidence=0.0,
                group_id=snapshot.group_id,
                state_group_id=group_key,
                text=snapshot.text,
            )
            if False:
                yield None
            return
        if token not in self._pending_replies:
            self._cancel_timers_for_token(token)
            _clear_qianji_extra(event)
            state.last_decision = "ignore (0.00)：回复构建已被更新的直接请求抢占。"
            self._log_decision(
                event,
                action="ignore",
                will_call_llm=False,
                reason="回复构建已被更新的直接请求抢占。",
                confidence=0.0,
                group_id=snapshot.group_id,
                state_group_id=group_key,
                text=snapshot.text,
            )
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
                "confidence": final_confidence,
                "reason": final_reason,
                "is_direct": wants_bot,
                "had_send_before": had_send_before,
            },
        )
        _set_extra(event, "enable_streaming", False)
        self._pending_replies[token] = PendingReply(
            group_id=group_key,
            token=token,
            confidence=final_confidence,
            reason=final_reason,
            is_direct=wants_bot,
            started_at=time.time(),
            request_id=request_id,
            status="starting",
            previous_streaming=previous_streaming,
            had_send_before=had_send_before,
        )
        self._sync_protected_context_groups()
        self._pending_events[token] = event
        state.append_user_message(decision_snapshot)
        self._schedule_pending_expiry(token, replace=True)
        _suppress_default_llm(event)
        state.last_decision = f"reply ({final_confidence:.2f})：{final_reason}"
        self._log_decision(
            event,
            action="reply",
            will_call_llm=True,
            reason=final_reason,
            confidence=final_confidence,
            group_id=snapshot.group_id,
            state_group_id=group_key,
            text=snapshot.text,
            request_id=request_id,
            timing_gate_called=timing_gate_called,
        )
        try:
            yield reply_request
        finally:
            _restore_streaming_extra(event, previous_streaming)
            pending_after_agent = self._pending_replies.get(token)
            if pending_after_agent is not None and pending_after_agent.status == "starting":
                self._pending_replies.pop(token, None)
                self._sync_protected_context_groups()
                self._cancel_timers_for_token(token)
                self._cleanup_pending_event(token, pending_after_agent)
                _clear_qianji_extra(event)
                state.last_decision = "ignore (0.00)：AstrBot 未确认 LLM 请求，已清理 pending。"
                self._log_lifecycle(
                    "LLM请求未确认",
                    event,
                    group_id=group_key,
                    token=token,
                    request_id=request_id,
                    detail="实际调用LLM=否；AstrBot 未触发 on_llm_request，已清理 pending。",
                )
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
            if self._should_track_external_event(event):
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
        _disable_provider_request_tools(request)
        self._log_lifecycle(
            "LLM请求确认",
            event,
            group_id=group_id,
            token=token,
            request_id=request_id,
            detail="实际调用LLM=是；本插件请求已进入 AstrBot ProviderRequest；已关闭工具调用以避免读空气回复误触发工具。",
        )
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
        self._sync_protected_context_groups()

    def record_llm_response(self, event: Any, response: Any) -> None:
        payload = _get_pending_payload(event)
        group_id = str(payload.get("group_id", "") or "")
        token = str(payload.get("token", "") or "")
        if not group_id or not token:
            if self._should_track_external_event(event):
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
            self._log_lifecycle(
                "LLM响应错误",
                event,
                group_id=group_id,
                token=token,
                request_id=pending.request_id,
                detail="模型响应 role=err，等待发送阶段收尾。",
            )
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
            self._log_lifecycle(
                "LLM空响应",
                event,
                group_id=group_id,
                token=token,
                request_id=pending.request_id,
                detail="模型未生成可见文本回复。",
            )
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
        self._log_lifecycle(
            "LLM响应完成",
            event,
            group_id=group_id,
            token=token,
            request_id=pending.request_id,
            detail=f"模型已产出可见文本，长度={len(reply_text)}。",
        )
        self._schedule_send_confirmation_expiry(token)

    def record_after_message_sent(self, event: Any) -> None:
        payload = _get_pending_payload(event)
        group_id = str(payload.get("group_id", "") or "")
        token = str(payload.get("token", "") or "")
        if not group_id or not token:
            if not self._should_track_external_event(event):
                _delete_extra(event, EXTERNAL_LLM_EXTRA_KEY)
                return
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
            self._sync_protected_context_groups()
            self._log_lifecycle(
                "发送确认忽略",
                event,
                group_id=group_id,
                token=token,
                detail="发送钩子中的 pending 已不存在或群 key 不匹配。",
            )
            return
        if pending.status not in {"active", "responded", "empty_response", "error_response", "timed_out"}:
            self._sync_protected_context_groups()
            self._log_lifecycle(
                "发送确认忽略",
                event,
                group_id=group_id,
                token=token,
                detail=f"pending 状态 {pending.status} 不需要发送收尾。",
            )
            return

        state = self.context_store.peek_group(group_id)
        if state is not None:
            state.touch()
        self._sync_protected_context_groups()
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
            self._log_lifecycle(
                "发送未确认",
                event,
                group_id=group_id,
                token=token,
                request_id=pending.request_id,
                detail=f"未确认本插件请求后发出消息，pending_status={pending.status}。",
            )
            return
        if (
            pending.status in {"active", "error_response"}
            or pending.status == "timed_out"
            or _looks_like_llm_error_text(sent_text)
        ):
            state.mark_bot_attempt(time.time())
            state.last_decision = "ignore (0.00)：AstrBot 会话模型未返回正常响应，已进入冷却但不写入群聊上下文。"
            self._log_lifecycle(
                "发送异常",
                event,
                group_id=group_id,
                token=token,
                request_id=pending.request_id,
                detail=f"发送文本像错误或 pending_status={pending.status}，不写入群聊上下文。",
            )
            return
        if not sent_text:
            sent_text = pending.response_text
        if not sent_text:
            state.mark_bot_attempt(time.time())
            state.last_decision = "ignore (0.00)：发送后未发现可记录的文本回复，已进入冷却。"
            self._log_lifecycle(
                "发送空文本",
                event,
                group_id=group_id,
                token=token,
                request_id=pending.request_id,
                detail="发送钩子触发但未找到可记录文本。",
            )
            return
        state.append_bot_reply(sent_text, time.time())
        state.last_decision = f"reply ({pending.confidence:.2f})：{pending.reason}"
        self._log_lifecycle(
            "发送确认",
            event,
            group_id=group_id,
            token=token,
            request_id=pending.request_id,
            detail=f"已记录 bot 回复，长度={len(sent_text)}。",
        )

    def _should_take_over_event(self, event: Any, snapshot: Any) -> bool:
        return bool(
            self.config.takeover_explicit_mentions
            and (snapshot.is_explicit_to_bot or getattr(event, "is_at_or_wake_command", False))
        )

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
                    self._sync_protected_context_groups()
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
            self._sync_protected_context_groups()
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
            event = self._pending_events.get(token)
            if event is not None:
                self._log_lifecycle(
                    "LLM等待超时",
                    event,
                    group_id=pending.group_id,
                    token=token,
                    request_id=pending.request_id,
                    detail="LLM 响应超过回复生成兜底时间，继续等待晚到发送确认。",
                )
            self._schedule_pending_expiry(token, replace=True)
            return
        if pending.status in {"responded", "empty_response", "error_response"}:
            self._finalize_unconfirmed_response(token, pending)
            return
        event = self._pending_events.get(token)
        self._pending_replies.pop(token, None)
        self._sync_protected_context_groups()
        self._cancel_timers_for_token(token)
        self._cleanup_pending_event(token, pending)
        if state is not None:
            state.last_decision = "ignore (0.00)：回复状态超时，已清理 pending。"
        if event is not None:
            self._log_lifecycle(
                "LLM状态超时清理",
                event,
                group_id=pending.group_id,
                token=token,
                request_id=pending.request_id,
                detail=f"pending_status={pending.status} 已超时清理。",
            )

    def _finalize_unconfirmed_response(self, token: str, pending: PendingReply) -> None:
        event = self._pending_events.get(token)
        self._pending_replies.pop(token, None)
        self._cancel_timers_for_token(token)
        self._cleanup_pending_event(token, pending)
        state = self.context_store.peek_group(pending.group_id)
        if state is not None:
            state.touch()
        self._sync_protected_context_groups()
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
        if event is not None:
            self._log_lifecycle(
                "发送确认超时清理",
                event,
                group_id=pending.group_id,
                token=token,
                request_id=pending.request_id,
                detail=f"pending_status={pending.status}，发送钩子未触发。",
            )

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
        if not self._should_track_external_event(event):
            return
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
        self._log_lifecycle(
            "外部发送记录",
            event,
            group_id=group_id,
            detail=f"其他链路已发送消息，长度={len(sent_text)}。",
        )

    def record_external_llm_request(self, event: Any) -> None:
        if not self._should_track_external_event(event):
            return
        if _should_ignore_external_event(event):
            return
        group_id = _state_group_id(event, _get_group_id(event))
        state = self.context_store.peek_group(group_id)
        _set_extra(event, EXTERNAL_LLM_REQUEST_EXTRA_KEY, True)
        if state is not None:
            state.mark_bot_attempt(time.time())
        self._log_lifecycle(
            "外部LLM请求",
            event,
            group_id=group_id,
            detail="同群其他链路请求 LLM，本插件记录冷却并避让。",
        )

    def record_external_llm_response(self, event: Any, response: Any) -> None:
        if not self._should_track_external_event(event):
            return
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
            self._log_lifecycle(
                "外部LLM响应",
                event,
                group_id=group_id,
                detail=f"其他链路 LLM 响应 role={role or 'normal'}，文本长度={len(reply_text)}。",
            )

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
        self._sync_protected_context_groups()
        self._pending_events.clear()

    def render_status(self, event: Any) -> str:
        group_id = _get_group_id(event)
        if not group_id:
            return "千机聆阙：请在群聊中使用这个指令。"
        if not _is_supported_platform(event):
            return "千机聆阙：当前只支持 aiocqhttp 群聊；本平台不会被被动监听。"
        state_group_id = _state_group_id(event, group_id)
        enabled = _is_group_enabled(self.config, group_id, state_group_id)
        effective_mode = _effective_mode(self.config, group_id, state_group_id)
        return (
            "千机聆阙状态\n"
            f"当前群：{group_id or '未知'}\n"
            f"总开关：{'开启' if self.config.enabled else '关闭'}\n"
            f"群开关：{'开启' if enabled else '关闭'}\n"
            f"模式：{mode_label(effective_mode)}\n"
            f"灰区节奏判断：{'开启' if self.config.llm_gate_enabled else '关闭'}"
        )

    def enable_group(self, event: Any) -> str:
        if not _is_supported_platform(event):
            return "千机聆阙：当前只支持 aiocqhttp 群聊。"
        group_id = _get_group_id(event)
        config_group_id = _state_group_id(event, group_id)
        if not group_id:
            return "千机聆阙：请在群聊中使用这个指令。"
        if not self.config.enabled:
            return "千机聆阙：总开关已关闭，请先在 WebUI 启用插件。"
        for candidate in _group_key_candidates(
            event,
            group_id,
            config_group_id,
            include_bare=not _is_scoped_group_key(config_group_id),
        ):
            if candidate in self.config.disabled_groups:
                self.config.disabled_groups.remove(candidate)
        if (
            not self.config.enables_all_groups()
            and config_group_id not in self.config.enabled_groups
        ):
            self.config.enabled_groups.append(config_group_id)
        self.config.save()
        if not _is_group_enabled(self.config, group_id, config_group_id):
            return "千机聆阙：已写入当前实例启用，但裸群号禁用安全阀仍生效；请在 WebUI 移除对应禁用项后再试。"
        return "千机聆阙：已开启当前群。"

    def disable_group(self, event: Any) -> str:
        if not _is_supported_platform(event):
            return "千机聆阙：当前只支持 aiocqhttp 群聊。"
        group_id = _get_group_id(event)
        config_group_id = _state_group_id(event, group_id)
        if not group_id:
            return "千机聆阙：请在群聊中使用这个指令。"
        if config_group_id not in self.config.disabled_groups:
            self.config.disabled_groups.append(config_group_id)
        self.config.save()
        return "千机聆阙：已关闭当前群。"

    def set_mode(self, event: Any, mode: str) -> str:
        if not _is_supported_platform(event):
            return "千机聆阙：当前只支持 aiocqhttp 群聊。"
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

    def _should_track_external_event(self, event: Any) -> bool:
        return _is_trackable_group_event(self.config, event)

    def _sync_protected_context_groups(self) -> None:
        max_groups = self.config.max_tracked_groups
        protected_group_ids: set[str] = set()
        if max_groups > 0:
            for pending in sorted(
                self._pending_replies.values(),
                key=lambda item: item.started_at,
                reverse=True,
            ):
                if pending.group_id:
                    protected_group_ids.add(pending.group_id)
                if len(protected_group_ids) >= max_groups:
                    break
        self.context_store.set_protected_groups(protected_group_ids)

    def _log_decision(
        self,
        event: Any,
        *,
        action: str,
        will_call_llm: bool,
        reason: str,
        confidence: float | None = None,
        group_id: str = "",
        state_group_id: str = "",
        text: str = "",
        request_id: int = 0,
        timing_gate_called: bool | None = None,
    ) -> None:
        if not self.config.log_decisions_enabled:
            return
        group_id = group_id or _get_group_id(event) or "未知"
        state_group_id = state_group_id or group_id
        score = "无" if confidence is None else f"{confidence:.2f}"
        extra = f" request_id={request_id}" if request_id else ""
        timing_extra = (
            ""
            if timing_gate_called is None
            else f" 节奏LLM实际={'是' if timing_gate_called else '否'}"
        )
        logger.info(
            "[千机聆阙] 判定 动作=%s 计划正式回复LLM=%s 分数=%s 群=%s 状态key=%s 平台=%s 发送者=%s 原因=%s 消息=%s%s%s",
            action,
            "是" if will_call_llm else "否",
            score,
            group_id,
            state_group_id,
            _event_platform_id(event) or _event_platform_name(event) or "未知",
            _event_sender_id(event) or "未知",
            _compact_log_text(reason, 160),
            self._log_message_summary(text or _message_text_from_event(event)),
            timing_extra,
            extra,
        )

    def _log_lifecycle(
        self,
        stage: str,
        event: Any,
        *,
        group_id: str = "",
        token: str = "",
        request_id: int = 0,
        detail: str = "",
    ) -> None:
        if not self.config.log_decisions_enabled:
            return
        logger.info(
            "[千机聆阙] %s 群=%s 平台=%s 发送者=%s token=%s request_id=%s 详情=%s",
            stage,
            group_id or _get_group_id(event) or "未知",
            _event_platform_id(event) or _event_platform_name(event) or "未知",
            _event_sender_id(event) or "未知",
            token or "无",
            request_id or "无",
            _compact_log_text(detail, 200),
        )

    def _log_message_summary(self, text: str) -> str:
        compact = " ".join(str(text or "").split())
        if self.config.log_message_excerpt_enabled:
            return _compact_log_text(compact, 80)
        return f"已隐藏(长度={len(compact)})"


def _get_group_id(event: Any) -> str:
    return group_id_from_event(event)


def _event_sender_id(event: Any) -> str:
    getter = getattr(event, "get_sender_id", None)
    if callable(getter):
        try:
            return str(getter() or "").strip()
        except Exception:
            return ""
    return str(getattr(event, "sender_id", "") or "").strip()


def _compact_log_text(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


def _event_unified_msg_origin(event: Any) -> str:
    return str(getattr(event, "unified_msg_origin", "") or "").strip()


def _state_group_id(event: Any, group_id: str) -> str:
    if not group_id:
        return ""
    unified_origin = _event_unified_msg_origin(event)
    if _is_valid_group_unified_origin(unified_origin, group_id, event):
        return unified_origin
    platform_id = _event_platform_id(event)
    platform_name = _event_platform_name(event)
    if platform_id:
        return f"{platform_id}:{group_id}"
    if platform_name:
        return f"{platform_name}:{group_id}"
    return group_id


def _legacy_platform_group_id(event: Any | None, group_id: str) -> str:
    if not group_id:
        return ""
    platform_id = _event_platform_id(event) if event is not None else ""
    platform_name = _event_platform_name(event) if event is not None else ""
    platform = platform_id or platform_name
    return f"{platform}:{group_id}" if platform else ""


def _group_key_candidates(
    event: Any | None,
    group_id: str,
    state_group_id: str = "",
    *,
    include_bare: bool = True,
) -> list[str]:
    candidates: list[str] = []
    raw_candidates = [
        state_group_id,
        _legacy_platform_group_id(event, group_id),
        _legacy_key_from_state_group_id(state_group_id, group_id),
    ]
    if include_bare:
        raw_candidates.append(group_id)
    for candidate in raw_candidates:
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _legacy_key_from_state_group_id(state_group_id: str, group_id: str) -> str:
    if not state_group_id or not group_id:
        return ""
    marker = ":GroupMessage:"
    if marker in state_group_id and state_group_id.endswith(group_id):
        platform_id = state_group_id.split(":", 1)[0]
        return f"{platform_id}:{group_id}" if platform_id else ""
    return ""


def _is_valid_group_unified_origin(unified_origin: str, group_id: str, event: Any | None = None) -> bool:
    if not unified_origin or not group_id:
        return False
    parts = unified_origin.split(":", 2)
    if len(parts) != 3 or not parts[0] or parts[1] != "GroupMessage" or parts[2] != group_id:
        return False
    platform_id = _event_platform_id(event) if event is not None else ""
    return not platform_id or parts[0] == platform_id


def _is_scoped_group_key(value: str) -> bool:
    return ":" in value


def _is_group_enabled(config: PluginConfig, group_id: str, state_group_id: str) -> bool:
    if not config.enabled:
        return False
    disabled_candidates = _group_key_candidates(None, group_id, state_group_id, include_bare=True)
    if any(candidate in config.disabled_groups for candidate in disabled_candidates):
        return False
    if config.enables_all_groups():
        return True
    if not config.enabled_groups:
        return False
    enabled_candidates = _group_key_candidates(
        None,
        group_id,
        state_group_id,
        include_bare=not _is_scoped_group_key(state_group_id),
    )
    return any(candidate in config.enabled_groups for candidate in enabled_candidates)


def _is_trackable_group_event(config: PluginConfig, event: Any) -> bool:
    if not _is_supported_platform(event):
        return False
    group_id = _get_group_id(event)
    if not group_id:
        return False
    return _is_group_enabled(config, group_id, _state_group_id(event, group_id))


def _effective_mode(config: PluginConfig, group_id: str, state_group_id: str) -> str:
    for candidate in _group_key_candidates(
        None,
        group_id,
        state_group_id,
        include_bare=not _is_scoped_group_key(state_group_id),
    ):
        if candidate in config.group_modes:
            return config.group_modes[candidate]
    return config.mode


def _decision_config_for_group(config: PluginConfig, group_id: str, state_group_id: str) -> PluginConfig:
    if group_id == state_group_id:
        return config
    return replace(
        config,
        enabled_groups=_mirror_group_entries(config.enabled_groups, group_id, state_group_id),
        disabled_groups=_mirror_group_entries(config.disabled_groups, group_id, state_group_id),
        group_modes={state_group_id: _effective_mode(config, group_id, state_group_id)},
        source=None,
    )


def _mirror_group_entries(values: list[str], group_id: str, state_group_id: str) -> list[str]:
    mirrored = list(values)
    candidates = _group_key_candidates(None, group_id, state_group_id, include_bare=True)
    if any(candidate in values for candidate in candidates) and state_group_id not in mirrored:
        mirrored.append(state_group_id)
    if state_group_id in values and group_id not in mirrored:
        mirrored.append(group_id)
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


def _timing_gate_should_fallback(
    result: Any,
    decision: Any,
    snapshot: Any,
    config: PluginConfig,
) -> bool:
    action = str(getattr(result, "action", "") or "")
    if not getattr(result, "called_llm", False):
        return True
    reason = str(getattr(result, "reason", "") or "")
    if any(marker in reason for marker in ("超时", "失败", "无效", "非 JSON")):
        return True
    confidence = _safe_float(getattr(result, "confidence", None), decision.confidence)
    low_confidence_ceiling = max(0.55, min(0.72, config.llm_gate_fallback_score + 0.25))
    return action in {"wait", "ignore"} and confidence <= low_confidence_ceiling


def _gray_message_is_worth_llm_gate(decision: Any, snapshot: Any, config: PluginConfig) -> bool:
    if not config.llm_gate_enabled:
        return False
    if _gray_local_fallback_should_reply(decision, snapshot, config):
        return True
    if _gray_weak_question_should_use_llm_gate(decision, snapshot, config):
        return True
    span = max(config.score_threshold_reply - config.score_threshold_ignore, 0.01)
    gray_position = (decision.confidence - config.score_threshold_ignore) / span
    return gray_position >= 0.65


def _gray_local_fallback_should_reply(decision: Any, snapshot: Any, config: PluginConfig) -> bool:
    if snapshot.is_direct_to_bot:
        return True
    text = snapshot.text.strip()
    if _looks_like_casual_non_question(text):
        return False
    if not _has_strong_reply_signal(text):
        return False
    return decision.confidence >= config.llm_gate_fallback_score


def _gray_weak_question_should_use_llm_gate(decision: Any, snapshot: Any, config: PluginConfig) -> bool:
    text = snapshot.text.strip()
    if _looks_like_casual_non_question(text):
        return False
    if not _looks_like_question_signal(text):
        return False
    return decision.confidence >= config.llm_gate_fallback_score


def _has_strong_reply_signal(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    strong_markers = (
        "帮我",
        "帮忙",
        "求助",
        "帮看",
        "帮看看",
        "帮我看",
        "帮忙看",
        "怎么办",
        "咋办",
        "如何",
        "怎么",
        "为什么",
        "什么",
        "啥",
        "咋回事",
        "咋弄",
        "咋整",
        "哪里",
        "哪儿",
        "哪个",
        "哪种",
        "哪位",
        "哪边",
        "在哪",
        "能不能",
        "可不可以",
        "行不行",
        "有办法",
    )
    if any(marker in stripped for marker in strong_markers):
        return True
    return (
        _looks_like_contextual_followup_question(stripped)
        or _looks_like_structured_what_question(stripped)
        or _looks_like_take_a_look_request(stripped)
    )


def _looks_like_question_signal(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 48:
        return False
    weak_fillers = {"是么", "是吗", "那呢", "啊", "哦", "嗯", "额"}
    if stripped in weak_fillers:
        return False
    if any(marker in stripped for marker in ("?", "？")):
        return True
    strong_question_markers = ("是否", "是不是", "有没有", "要不要", "该不该", "会不会", "需不需要", "能否")
    if any(marker in stripped for marker in strong_question_markers):
        return True
    if stripped.endswith(("吗", "么")):
        return True
    if _looks_like_contextual_followup_question(stripped):
        return True
    return False


def _looks_like_contextual_followup_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 8:
        return False
    if not stripped.endswith(("呢", "吗", "么", "？", "?")):
        return False
    return stripped.startswith(("这", "那", "这个", "那个", "它", "他", "她"))


def _looks_like_structured_what_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 24:
        return False
    markers = (
        "是什么",
        "什么是",
        "啥是",
        "是啥",
        "什么意思",
        "什么情况",
        "什么原因",
        "什么问题",
        "什么办法",
        "为啥",
    )
    return any(marker in stripped for marker in markers)


def _looks_like_take_a_look_request(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 24:
        return False
    if stripped.startswith(("我看下", "我看一下", "我看看")):
        return False
    return stripped.startswith(("看下", "看一下", "看看")) or any(
        marker in stripped for marker in ("你看下", "你看一下", "帮我看看", "帮忙看看")
    )


def _looks_like_casual_non_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    casual_exact = {"没什么", "一起玩吗", "好呢", "行呢", "可以呢", "是吗", "是么", "那呢"}
    if stripped in casual_exact:
        return True
    casual_prefixes = ("没什么", "没啥", "没事", "无所谓")
    if stripped.startswith(casual_prefixes):
        return True
    casual_markers = ("没什么", "没啥", "有什么好", "有啥好", "一起玩吗")
    if any(marker in stripped for marker in casual_markers):
        return True
    if len(stripped) <= 8 and any(stripped.startswith(prefix) for prefix in ("好呢", "行呢", "可以呢")):
        return True
    return False


def _local_fallback_confidence(decision: Any) -> float:
    confidence = _safe_float(getattr(decision, "confidence", None), 0.0)
    return max(confidence, min(0.72, confidence + 0.16))


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


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
    _disable_provider_request_tools(request)
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


def _disable_provider_request_tools(request: Any) -> None:
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


def _strip_tool_prompt_text(text: str) -> str:
    lines: list[str] = []
    raw_lines = text.splitlines()
    index = 0
    while index < len(raw_lines):
        line = raw_lines[index]
        stripped = line.strip()
        if stripped.lower() == "## skills" and _looks_like_astrbot_skills_block(raw_lines, index):
            index = _astrbot_skills_block_end(raw_lines, index)
            continue
        if _is_tool_prompt_line(stripped):
            index = _skip_sdk_workspace_block(raw_lines, index + 1)
            continue
        lines.append(line)
        index += 1
    return "\n".join(lines).strip()


def _looks_like_astrbot_skills_block(lines: list[str], start_index: int) -> bool:
    window = "\n".join(line.strip() for line in lines[start_index + 1 : start_index + 12])
    lowered = window.lower()
    return (
        "you have specialized skills" in lowered
        and "`skill.md`" in lowered
        and (
            "### available skills" in lowered
            or "### skill rules" in lowered
            or "**failure handling**" in lowered
        )
    )


def _astrbot_skills_block_end(lines: list[str], start_index: int) -> int:
    rules_start = _actual_skill_rules_start(lines, start_index)
    if rules_start is not None:
        end_index = _actual_skill_rules_end(lines, rules_start, len(lines))
        if end_index is not None:
            return end_index
    tool_prompt_index = _next_tool_prompt_index(lines, start_index + 1)
    if tool_prompt_index is not None:
        return tool_prompt_index
    return len(lines)


def _actual_skill_rules_start(lines: list[str], start_index: int) -> int | None:
    for index in range(start_index + 1, len(lines)):
        if lines[index].strip().lower() != "### skill rules":
            continue
        if not _has_skill_file_anchor(lines, start_index, index):
            continue
        if _actual_skill_rules_end(lines, index, len(lines)) is not None:
            return index
    return None


def _has_skill_file_anchor(lines: list[str], start_index: int, rules_start: int) -> bool:
    for index in range(rules_start - 1, start_index, -1):
        stripped = lines[index].strip()
        if _is_tool_prompt_line(stripped):
            return False
        if stripped.startswith("File: `"):
            return True
    return False


def _actual_skill_rules_end(lines: list[str], rules_start: int, end_index: int) -> int | None:
    expected: list[tuple[str, str]] = [
        ("1. **Discovery**", "complete skill inventory"),
        ("2. **When to trigger**", "Use a skill"),
        ("3. **Mandatory grounding**", "first read its `SKILL.md`"),
        ("4. **Progressive disclosure**", "Load only what is directly"),
        ("5. **Coordination**", "multiple skills apply"),
        ("6. **Context hygiene**", "Avoid deep reference chasing"),
        ("7. **Failure handling**", "continue with the best alternative"),
    ]
    cursor = rules_start + 1
    last_match: int | None = None
    for prefix, marker in expected:
        matched_index = _find_skill_rule_line(lines, cursor, end_index, prefix, marker)
        if matched_index is None:
            return None
        last_match = matched_index
        cursor = matched_index + 1
    return last_match + 1 if last_match is not None else None


def _find_skill_rule_line(
    lines: list[str],
    start_index: int,
    end_index: int,
    prefix: str,
    marker: str,
) -> int | None:
    for index in range(start_index, end_index):
        stripped = lines[index].strip()
        if stripped.startswith(prefix) and marker in stripped:
            return index
    return None


def _next_nonempty_line_index(lines: list[str], start_index: int, end_index: int) -> int | None:
    for index in range(start_index, end_index):
        if lines[index].strip():
            return index
    return None


def _next_tool_prompt_index(lines: list[str], start_index: int) -> int | None:
    for index in range(start_index, len(lines)):
        if _is_tool_prompt_line(lines[index].strip()):
            return index
    return None


def _is_tool_prompt_line(stripped: str) -> bool:
    if stripped in SDK_TOOL_PROMPTS:
        return True
    return not SDK_TOOL_PROMPTS and stripped in KNOWN_TOOL_PROMPTS


def _is_workspace_prompt_line(stripped: str) -> bool:
    if not stripped.startswith("Current workspace you can use: `"):
        return False
    if not stripped.endswith("`"):
        return False
    workspace_path = stripped.removeprefix("Current workspace you can use: `").removesuffix("`")
    return _is_astrbot_workspace_path(workspace_path)


def _is_workspace_followup_line(stripped: str) -> bool:
    return (
        stripped
        == "Unless the user explicitly specifies a different directory, perform all file-related operations in this workspace."
    )


def _skip_sdk_workspace_block(lines: list[str], start_index: int) -> int:
    if start_index >= len(lines):
        return start_index
    if not _is_workspace_prompt_line(lines[start_index].strip()):
        return start_index
    followup_index = start_index + 1
    if followup_index >= len(lines) or not _is_workspace_followup_line(lines[followup_index].strip()):
        return start_index
    return followup_index + 1


def _is_astrbot_workspace_path(path: str) -> bool:
    if not ASTRBOT_WORKSPACES_PATH:
        return False
    try:
        normalized_path = os.path.realpath(path)
        common_path = os.path.commonpath([ASTRBOT_WORKSPACES_PATH, normalized_path])
    except (OSError, ValueError):
        return False
    return common_path == ASTRBOT_WORKSPACES_PATH


def _get_event_result(event: Any) -> Any:
    result_getter = getattr(event, "get_result", None)
    return result_getter() if callable(result_getter) else getattr(event, "_result", None)


def _extract_sent_text(event: Any) -> str:
    return extract_chain_text(_get_event_result(event))


def _event_has_send_operation(event: Any) -> bool:
    return bool(getattr(event, "_has_send_oper", False))
