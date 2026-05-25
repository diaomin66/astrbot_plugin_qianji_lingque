from __future__ import annotations

import json
import os
import re
import base64
import binascii
from typing import Any
from urllib.request import url2pathname
from urllib.parse import urlparse

from .context import GroupState
from .event_utils import MessageSnapshot
from .prompts import build_reply_prompt, build_reply_system_prompt

try:
    from astrbot.core.message.components import At, Image, Plain, Record, Reply
except Exception:
    At = Image = Plain = Record = Reply = None

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
except Exception:
    get_astrbot_temp_path = None


ASTRBOT_MEDIA_COMPONENTS = tuple(
    component for component in (Plain, At, Image, Record, Reply) if isinstance(component, type)
)

MAX_BASE64_MEDIA_LENGTH = 700_000
MAX_LOCAL_MEDIA_BYTES = 1_500_000


class LLMClient:
    def __init__(self, context: Any) -> None:
        self.context = context
        self.last_error: str = ""

    def extract_visible_reply(self, response: Any) -> str:
        return sanitize_reply(extract_completion_text(response))

    async def build_reply_request(
        self,
        event: Any,
        snapshot: MessageSnapshot,
        state: GroupState,
        decision_reason: str,
    ) -> Any:
        request_llm = getattr(event, "request_llm", None)
        if not callable(request_llm):
            self.last_error = "当前 AstrBot 事件不支持 request_llm。"
            return None
        if not await self._provider_request_supported(event):
            return None
        self.last_error = ""
        media_urls = await _collect_media_urls(event)
        if media_urls is None:
            self.last_error = "消息包含复杂媒体或引用附件，交给 AstrBot 默认链路。"
            return None
        conversation = await self._current_conversation(event)
        if conversation is None:
            return None
        image_urls, audio_urls = media_urls
        return request_llm(
            prompt=build_reply_prompt(snapshot),
            system_prompt=build_reply_system_prompt(snapshot, state, decision_reason),
            conversation=conversation,
            image_urls=image_urls,
            audio_urls=audio_urls,
        )

    async def _provider_request_supported(self, event: Any) -> bool:
        agent_runner_type = self._agent_runner_type(event)
        if agent_runner_type and agent_runner_type != "local":
            self.last_error = f"当前 Agent runner（{agent_runner_type}）不适合插件主动 ProviderRequest，交给默认链路。"
            return False
        manager = getattr(self.context, "provider_manager", None)
        if manager is None:
            return True
        provider_getter = getattr(manager, "get_provider_by_id", None)
        current_getter = getattr(self.context, "get_current_chat_provider_id", None)
        if not callable(provider_getter) or not callable(current_getter):
            return True
        try:
            provider_id = await current_getter(umo=str(getattr(event, "unified_msg_origin", "") or ""))
            provider = await provider_getter(provider_id)
        except Exception as exc:
            self.last_error = f"检查 AstrBot provider 失败：{exc.__class__.__name__}"
            return False
        provider_type = str(getattr(provider, "provider_config", {}).get("type", "") or "").lower()
        third_party_runners = {"coze", "dify", "dashscope", "deerflow"}
        if provider_type in third_party_runners:
            self.last_error = f"当前 provider runner（{provider_type}）不适合插件主动 ProviderRequest，交给默认链路。"
            return False
        return True

    def _agent_runner_type(self, event: Any) -> str:
        config_getter = getattr(self.context, "get_config", None)
        if not callable(config_getter):
            return ""
        try:
            config = config_getter(umo=str(getattr(event, "unified_msg_origin", "") or ""))
        except TypeError:
            config = config_getter()
        except Exception:
            return ""
        provider_settings = {}
        if isinstance(config, dict):
            provider_settings = config.get("provider_settings", {}) or {}
        else:
            getter = getattr(config, "get", None)
            if callable(getter):
                provider_settings = getter("provider_settings", {}) or {}
        runner_type = str(provider_settings.get("agent_runner_type", "") or "").strip().lower()
        return runner_type or "local"

    async def _current_conversation(self, event: Any) -> Any:
        manager = getattr(self.context, "conversation_manager", None)
        if manager is None:
            self.last_error = "未找到 conversation_manager。"
            return None
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        get_current = getattr(manager, "get_curr_conversation_id", None)
        new_conversation = getattr(manager, "new_conversation", None)
        get_conversation = getattr(manager, "get_conversation", None)
        if not callable(get_current) or not callable(new_conversation) or not callable(get_conversation):
            self.last_error = "conversation_manager 缺少必要方法。"
            return None
        platform_id = _event_platform_id(event)
        try:
            conversation_id = await get_current(umo)
            if not conversation_id:
                conversation_id = await new_conversation(umo, platform_id)
            conversation = await get_conversation(umo, conversation_id)
            if conversation is not None:
                return conversation
            conversation_id = await new_conversation(umo, platform_id)
            return await get_conversation(umo, conversation_id)
        except Exception as exc:
            self.last_error = f"获取 AstrBot 会话失败：{exc.__class__.__name__}"
            return None

def extract_completion_text(response: Any) -> str:
    if str(getattr(response, "role", "") or "").lower() == "err":
        return ""
    for attr in ("completion_text", "text", "content"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    chain_text = extract_chain_text(getattr(response, "result_chain", None))
    if chain_text:
        return chain_text
    if isinstance(response, str):
        return response.strip()
    return ""


def extract_chain_text(chain_holder: Any) -> str:
    if chain_holder is None:
        return ""
    plain_getter = getattr(chain_holder, "get_plain_text", None)
    if callable(plain_getter):
        try:
            value = plain_getter()
        except Exception:
            value = ""
        if isinstance(value, str) and value.strip():
            return value.strip()
    chain = getattr(chain_holder, "chain", chain_holder)
    if not isinstance(chain, list):
        return ""
    parts: list[str] = []
    for component in chain:
        if component.__class__.__name__ != "Plain":
            continue
        value = getattr(component, "text", None)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts).strip()


def sanitize_reply(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json|text)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = cleaned.strip("\"'“”")
    if _looks_like_structured_or_tool_output(cleaned):
        cleaned = _extract_safe_json_reply(cleaned)
    return cleaned[:500].strip()


def _looks_like_structured_or_tool_output(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    if lowered.startswith(("{", "[")):
        return True
    markers = ("[tool]", "<tool", "<reasoning", "reasoning:", "tool_call")
    return any(marker in lowered for marker in markers)


def _extract_safe_json_reply(text: str) -> str:
    if not text.startswith("{"):
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    for key in ("reply", "text", "content", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _event_platform_id(event: Any) -> str:
    getter = getattr(event, "get_platform_id", None)
    if callable(getter):
        try:
            return str(getter() or "")
        except TypeError:
            return ""
    return str(getattr(event, "platform_id", "") or "")


async def _collect_media_urls(event: Any) -> tuple[list[str], list[str]] | None:
    messages = _event_messages(event)
    return await _collect_media_from_components(messages)


async def _collect_media_from_components(
    messages: list[Any],
    *,
    depth: int = 0,
) -> tuple[list[str], list[str]] | None:
    if depth > 2:
        return None
    image_urls: list[str] = []
    audio_urls: list[str] = []
    for component in messages:
        kind = _component_kind(component)
        if not kind:
            return None
        if kind == "image":
            value = await _component_file_or_url(component, kind)
            if not value:
                return None
            image_urls.append(value)
        elif kind == "record":
            value = await _component_file_or_url(component, kind)
            if not value:
                return None
            audio_urls.append(value)
        elif kind == "reply":
            nested = _reply_chain(component)
            if nested is None:
                return None
            nested_media = await _collect_media_from_components(nested, depth=depth + 1)
            if nested_media is None:
                return None
            nested_images, nested_audios = nested_media
            image_urls.extend(nested_images)
            audio_urls.extend(nested_audios)
    return _dedupe(image_urls), _dedupe(audio_urls)


def _component_kind(component: Any) -> str:
    if ASTRBOT_MEDIA_COMPONENTS and isinstance(component, ASTRBOT_MEDIA_COMPONENTS):
        if Image is not None and isinstance(component, Image):
            return "image"
        if Record is not None and isinstance(component, Record):
            return "record"
        if Reply is not None and isinstance(component, Reply):
            return "reply"
        if Plain is not None and isinstance(component, Plain):
            return "plain"
        if At is not None and isinstance(component, At):
            return "at"
    name = component.__class__.__name__
    return {
        "Plain": "plain",
        "At": "at",
        "Image": "image",
        "Record": "record",
        "Reply": "reply",
    }.get(name, "")


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


def _reply_chain(component: Any) -> list[Any] | None:
    chain = getattr(component, "chain", None)
    if chain is None:
        return []
    return chain if isinstance(chain, list) else None


async def _component_file_or_url(component: Any, media_kind: str) -> str:
    for attr in ("url", "file", "path"):
        value = getattr(component, attr, None)
        if isinstance(value, str) and value.strip():
            raw_value = value.strip()
            if _is_remote_media_ref(raw_value):
                return ""
            media_ref = _safe_media_ref(raw_value, media_kind)
            if media_ref:
                return media_ref
    converter = getattr(component, "convert_to_file_path", None)
    if callable(converter):
        try:
            value = await converter()
        except Exception:
            return ""
        if isinstance(value, str) and value.strip():
            return _safe_media_ref(value.strip(), media_kind)
    return ""


def _is_remote_media_ref(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _safe_media_ref(value: str, media_kind: str) -> str:
    if os.path.isabs(value) and _is_valid_local_media(value, media_kind):
        return value
    parsed = urlparse(value)
    if _is_remote_media_ref(value):
        return ""
    if parsed.scheme == "file" and parsed.path and not parsed.netloc:
        local_path = _file_uri_to_path(parsed.path)
        return local_path if local_path and _is_valid_local_media(local_path, media_kind) else ""
    if parsed.scheme == "base64" and _valid_base64_media(value, media_kind):
        return value
    if parsed.scheme:
        return ""
    return ""


def _file_uri_to_path(path: str) -> str:
    local_path = url2pathname(path)
    if re.match(r"^/[A-Za-z]:[/\\]", local_path):
        local_path = local_path[1:]
    return local_path


def _valid_base64_media(value: str, media_kind: str) -> bool:
    payload = value.removeprefix("base64://")
    if not payload or len(payload) > MAX_BASE64_MEDIA_LENGTH:
        return False
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return False
    if media_kind == "image":
        return _looks_like_jpeg_bytes(data)
    if media_kind == "record":
        return _looks_like_wav_bytes(data)
    return False


def _looks_like_jpeg_bytes(data: bytes) -> bool:
    return len(data) >= 20 and data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9")


def _looks_like_wav_bytes(data: bytes) -> bool:
    return (
        len(data) >= 12
        and data.startswith(b"RIFF")
        and data[8:12] == b"WAVE"
        and _riff_declared_length_matches(data, len(data))
    )


def _riff_declared_length_matches(header: bytes, decoded_length: int) -> bool:
    if len(header) < 8:
        return False
    declared_length = int.from_bytes(header[4:8], "little") + 8
    return declared_length == decoded_length


def _remote_media_extension_supported(path: str, media_kind: str) -> bool:
    suffix = os.path.splitext(path.lower())[1]
    if media_kind == "image":
        return suffix in {".jpg", ".jpeg", ".jfif"}
    if media_kind == "record":
        return suffix == ".wav"
    return False


def _is_valid_local_media(path: str, media_kind: str) -> bool:
    if not _is_under_allowed_temp(path):
        return False
    if not os.path.isfile(path):
        return False
    try:
        if os.path.getsize(path) > MAX_LOCAL_MEDIA_BYTES:
            return False
    except OSError:
        return False
    return _local_media_magic_supported(path, media_kind)


def _local_media_magic_supported(path: str, media_kind: str) -> bool:
    try:
        with open(path, "rb") as handle:
            if media_kind == "image":
                head = handle.read(3)
                handle.seek(-2, os.SEEK_END)
                tail = handle.read(2)
                return head == b"\xff\xd8\xff" and tail == b"\xff\xd9"
            if media_kind == "record":
                header = handle.read(12)
                file_size = os.path.getsize(path)
                return (
                    len(header) >= 12
                    and header.startswith(b"RIFF")
                    and header[8:12] == b"WAVE"
                    and _riff_declared_length_matches(header, file_size)
                )
    except (OSError, ValueError):
        return False
    return False


def _is_under_allowed_temp(path: str) -> bool:
    try:
        candidate = os.path.normcase(os.path.realpath(path))
    except OSError:
        return False
    roots: list[str] = []
    if callable(get_astrbot_temp_path):
        try:
            roots.append(str(get_astrbot_temp_path()))
        except Exception:
            pass
    elif temp_path := _lazy_astrbot_temp_path():
        roots.append(temp_path)
    for root in roots:
        try:
            root_path = os.path.normcase(os.path.realpath(root))
            common = os.path.commonpath([candidate, root_path])
        except (OSError, ValueError):
            continue
        if common == root_path:
            return True
    return False


def _lazy_astrbot_temp_path() -> str:
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_temp_path as getter
    except Exception:
        return ""
    try:
        return str(getter())
    except Exception:
        return ""


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
