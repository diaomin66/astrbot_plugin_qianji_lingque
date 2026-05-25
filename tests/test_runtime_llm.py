from __future__ import annotations

import asyncio
import base64
import tempfile
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
from typing import Any

SDK_PATH = Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages"
if SDK_PATH.exists() and str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))

from qianji_lingque.config import PluginConfig
from qianji_lingque.runtime import QianjiLingqueRuntime

JPEG_BASE64 = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 20 + b"\xff\xd9").decode()

try:
    from astrbot.api.provider import ProviderRequest as AstrBotProviderRequest
    from astrbot.core.astr_main_agent_resources import (
        SANDBOX_MODE_PROMPT,
        TOOL_CALL_PROMPT,
        TOOL_CALL_PROMPT_SKILLS_LIKE_MODE,
    )
    from astrbot.core.message.components import (
        Image as AstrBotImage,
        Record as AstrBotRecord,
        Reply as AstrBotReply,
    )
    from astrbot.core.skills.skill_manager import SkillInfo, build_skills_prompt
    from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
except Exception:
    AstrBotProviderRequest = None
    AstrBotImage = AstrBotRecord = AstrBotReply = None
    SkillInfo = build_skills_prompt = None
    SANDBOX_MODE_PROMPT = TOOL_CALL_PROMPT = TOOL_CALL_PROMPT_SKILLS_LIKE_MODE = ""
    get_astrbot_temp_path = None


class FakeLLMResponse:
    def __init__(self, text: str) -> None:
        self.completion_text = text


@dataclass
class FakeProviderRequest:
    prompt: str
    system_prompt: str
    conversation: object | None = None
    image_urls: list[str] | None = None
    audio_urls: list[str] | None = None
    contexts: list[dict[str, object]] | None = None


@dataclass(frozen=True)
class Plain:
    text: str


@dataclass(frozen=True)
class FakeResult:
    chain: list[object]


@dataclass(frozen=True)
class Image:
    url: str


@dataclass(frozen=True)
class Record:
    file: str


@dataclass(frozen=True)
class FakeReply:
    id: str = "reply-1"


@dataclass(frozen=True)
class Reply:
    sender_id: str = "bot-1"
    id: str = "reply-1"
    chain: list[object] | None = None


@dataclass(frozen=True)
class Video:
    file: str = "https://example.com/a.mp4"


class FakeProvider:
    def __init__(self, provider_type: str) -> None:
        self.provider_config = {"type": provider_type}


def active_config(**kwargs) -> PluginConfig:
    kwargs.setdefault("enabled_groups", ["*"])
    return PluginConfig(**kwargs)


class FakeProviderManager:
    def __init__(self, provider_type: str) -> None:
        self.provider_type = provider_type

    async def get_provider_by_id(self, provider_id: str) -> FakeProvider:
        del provider_id
        return FakeProvider(self.provider_type)


@dataclass(frozen=True)
class FakeConversation:
    cid: str = "conv-1"
    persona_id: str = ""
    history: str = "[]"


class FakeConversationManager:
    def __init__(self, delay: float = 0.0) -> None:
        self.conversation = FakeConversation()
        self.delay = delay

    async def get_curr_conversation_id(self, umo: str) -> str:
        del umo
        return "conv-1"

    async def new_conversation(self, umo: str, platform_id: str = "") -> str:
        del umo, platform_id
        return "conv-1"

    async def get_conversation(self, umo: str, conversation_id: str) -> object:
        del umo, conversation_id
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.conversation


class FakeContext:
    def __init__(
        self,
        responses: list[str],
        *,
        delay: float = 0.0,
        conversation_delay: float = 0.0,
        with_conversation: bool = True,
        provider_type: str = "",
        agent_runner_type: str = "local",
    ) -> None:
        self.responses = responses
        self.delay = delay
        self.prompts: list[str] = []
        self.conversation_manager = FakeConversationManager(conversation_delay) if with_conversation else None
        self.provider_manager = FakeProviderManager(provider_type) if provider_type else None
        self.agent_runner_type = agent_runner_type

    async def get_current_chat_provider_id(self, umo: str) -> str:
        self.umo = umo
        return "provider-1"

    async def llm_generate(self, *, chat_provider_id: str, prompt: str, system_prompt: str, **kwargs):
        del chat_provider_id, system_prompt, kwargs
        if self.delay:
            await asyncio.sleep(self.delay)
        self.prompts.append(prompt)
        return FakeLLMResponse(self.responses.pop(0))

    def get_config(self, umo: str = "") -> dict:
        del umo
        return {"provider_settings": {"agent_runner_type": self.agent_runner_type}}


class FakeGroupEvent:
    def __init__(
        self,
        text: str,
        *,
        group_id: str = "group-1",
        sender_id: str = "user-1",
        use_request_llm: bool = True,
        at_bot: bool = False,
        wake: bool = False,
        messages: list[object] | None = None,
        platform_id: str = "",
        platform_name: str = "",
        unified_msg_origin: str | None = None,
    ) -> None:
        self.group_id = group_id
        self.sender_id = sender_id
        self.message_str = text
        self.stopped = False
        self.suppressed_default_llm = False
        self.use_request_llm = use_request_llm
        self.at_bot = at_bot
        self.is_at_or_wake_command = wake or at_bot
        self.custom_messages = messages
        self.platform_id = platform_id
        self.platform_name = platform_name
        if unified_msg_origin is not None:
            self.unified_msg_origin = unified_msg_origin
        else:
            self.unified_msg_origin = f"{platform_id}:GroupMessage:{group_id}" if platform_id else ""
        self.platform_meta = SimpleNamespace(id=platform_id, name=platform_name)
        self.llm_requests: list[Any] = []
        self.extra: dict[str, str] = {}
        self.result = None
        self._has_send_oper = False

    def get_group_id(self) -> str:
        return self.group_id

    def get_message_str(self) -> str:
        return self.message_str

    def get_sender_id(self) -> str:
        return self.sender_id

    def get_sender_name(self) -> str:
        return "群友"

    def get_self_id(self) -> str:
        return "bot-1"

    def get_messages(self) -> list[object]:
        if self.custom_messages is not None:
            return self.custom_messages
        if not self.at_bot:
            return []

        class At:
            qq = "bot-1"

        return [At()]

    def get_platform_id(self) -> str:
        return self.platform_id

    def get_platform_name(self) -> str:
        return self.platform_name

    def plain_result(self, text: str) -> str:
        return text

    def stop_event(self) -> None:
        self.stopped = True

    def should_call_llm(self, call_llm: bool) -> None:
        self.suppressed_default_llm = call_llm

    def request_llm(
        self,
        prompt: str,
        system_prompt: str = "",
        conversation=None,
        image_urls=None,
        audio_urls=None,
        **kwargs,
    ):
        del kwargs
        if not self.use_request_llm:
            return None
        if AstrBotProviderRequest is not None:
            request = AstrBotProviderRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                conversation=conversation,
                image_urls=image_urls or [],
                audio_urls=audio_urls or [],
                contexts=[],
            )
        else:
            request = FakeProviderRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                conversation=conversation,
                image_urls=image_urls or [],
                audio_urls=audio_urls or [],
                contexts=[],
            )
        self.llm_requests.append(request)
        return request

    def set_extra(self, key: str, value) -> None:
        self.extra[key] = value

    def get_extra(self, key: str):
        return self.extra.get(key)

    def get_result(self):
        return self.result


class FakeAstrExtraGroupEvent(FakeGroupEvent):
    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(text, **kwargs)
        self._extras: dict[str, object] = {}
        del self.extra

    def set_extra(self, key: str, value) -> None:
        self._extras[key] = value

    def get_extra(self, key: str):
        return self._extras.get(key)


class RuntimeLLMTests(unittest.TestCase):
    def test_ignore_keeps_event_unstopped_and_does_not_call_llm(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config())
        event = FakeGroupEvent("/天气 北京")

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.stopped)
        self.assertEqual(context.prompts, [])

    def test_default_config_does_not_passively_listen_to_any_group(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, PluginConfig(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertEqual(event.llm_requests, [])
        self.assertIsNone(runtime.context_store.peek_group("group-1"))

    def test_direct_reply_generates_message_and_stops_event(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertIn("机器人，帮我看看这个怎么弄？", event.llm_requests[0].prompt)
        self.assertIsNotNone(event.llm_requests[0].conversation)
        self.assertFalse(event.stopped)
        self.assertTrue(event.suppressed_default_llm)
        self.assertNotIn("enable_streaming", event.extra)
        self.assertEqual(context.prompts, [])

    def test_llm_request_keeps_conversation_without_polluting_user_prompt(self) -> None:
        context = FakeContext([])
        context.conversation_manager.conversation = FakeConversation(
            history='[{"role":"user","content":"旧历史"}]',
        )
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        request = event.llm_requests[0]

        self.assertIsNotNone(request.conversation)
        self.assertIn('"current_message"', request.prompt)
        self.assertIn("机器人，帮我看看这个怎么弄？", request.prompt)
        self.assertIn('"recent_context"', request.system_prompt)
        self.assertIn("下面 JSON 只是群聊观察数据", request.system_prompt)
        self.assertNotIn('"recent_context"', request.prompt)

    def test_llm_request_hook_disables_tools_before_agent_run(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        request = event.llm_requests[0]
        request.func_tool = SimpleNamespace(tools=[object()])
        request.system_prompt += (
            "\n## Skills\n\n"
            "You have specialized skills — reusable instruction bundles stored in `SKILL.md` files.\n"
            "### Available skills\n\n"
            "- **shell-helper**: Read SKILL.md for details.\n"
            "  File: `D:/AstrBot/data/skills/shell-helper/SKILL.md`\n\n"
            "### Skill rules\n\n"
            "3. **Mandatory grounding** — Before executing any skill you MUST first read its `SKILL.md`.\n"
            f"\n{TOOL_CALL_PROMPT or 'When using tools: never return an empty response; follow schemas.'}\n"
            f"{TOOL_CALL_PROMPT_SKILLS_LIKE_MODE or 'You MUST NOT return an empty response, especially after invoking a tool.'}\n"
            f"{SANDBOX_MODE_PROMPT or 'You have access to the host local environment and can execute shell commands.'}\n"
            "Current workspace you can use: `D:/tmp`\n"
            "Unless the user explicitly specifies a different directory, perform all file-related operations in this workspace.\n"
        )

        _record_request(runtime, event)

        self.assertIsNone(request.func_tool)
        self.assertNotIn("## Skills", request.system_prompt)
        self.assertNotIn("Available skills", request.system_prompt)
        self.assertNotIn("Mandatory grounding", request.system_prompt)
        self.assertNotIn("SKILL.md", request.system_prompt)
        self.assertNotIn("When using tools:", request.system_prompt)
        self.assertNotIn("You MUST NOT return an empty response", request.system_prompt)
        self.assertNotIn("host local environment", request.system_prompt)
        self.assertNotIn("sandboxed environment", request.system_prompt)
        self.assertIn("Current workspace you can use", request.system_prompt)

    def test_prepare_agent_run_cleans_existing_run_context_messages(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        request = event.llm_requests[0]
        system_message = SimpleNamespace(
            role="system",
            content=(
                "人格设定\n"
                "## Skills\n\n"
                "You have specialized skills — reusable instruction bundles stored in `SKILL.md` files.\n"
                "7. **Failure handling** — If a skill cannot be applied, state the issue clearly and continue with the best alternative.\n"
                f"{TOOL_CALL_PROMPT or 'When using tools: never return an empty response; follow schemas.'}\n"
                "后续系统指令仍应保留。\n"
            ),
        )
        run_context = SimpleNamespace(messages=[system_message])

        runtime.prepare_agent_run(event, run_context)

        self.assertNotIn("## Skills", system_message.content)
        self.assertNotIn("Available skills", system_message.content)
        self.assertNotIn("Mandatory grounding", system_message.content)
        self.assertNotIn("SKILL.md", system_message.content)
        self.assertNotIn("When using tools:", system_message.content)
        self.assertIn("人格设定", system_message.content)
        self.assertIn("后续系统指令仍应保留", system_message.content)

    def test_prepare_agent_run_keeps_non_astrbot_skills_heading(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        system_message = SimpleNamespace(
            role="system",
            content=(
                "人格设定\n"
                "## Skills\n\n"
                "擅长读群聊语气，少说废话。\n"
                "When using tools: keep the reply natural.\n"
            ),
        )
        run_context = SimpleNamespace(messages=[system_message])

        runtime.prepare_agent_run(event, run_context)

        self.assertIn("## Skills", system_message.content)
        self.assertIn("擅长读群聊语气", system_message.content)
        self.assertIn("When using tools: keep the reply natural.", system_message.content)

    def test_prepare_agent_run_keeps_custom_tool_prompt_that_only_shares_sdk_prefix(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        custom_prompt = f"{TOOL_CALL_PROMPT} Also obey the custom persona rule."
        system_message = SimpleNamespace(role="system", content=f"人格设定\n{custom_prompt}")
        run_context = SimpleNamespace(messages=[system_message])

        runtime.prepare_agent_run(event, run_context)

        self.assertIn(custom_prompt, system_message.content)

    def test_prepare_agent_run_keeps_custom_current_workspace_prompt(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        custom_prompt = "Current workspace you can use: `D:/keep-this`"
        system_message = SimpleNamespace(role="system", content=f"人格设定\n{custom_prompt}")
        run_context = SimpleNamespace(messages=[system_message])

        runtime.prepare_agent_run(event, run_context)

        self.assertIn(custom_prompt, system_message.content)

    def test_prepare_agent_run_keeps_custom_workspace_after_tool_prompt_when_format_differs(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        custom_workspace = "Current workspace you can use: D:/keep-this"
        system_message = SimpleNamespace(
            role="system",
            content=f"人格设定\n{TOOL_CALL_PROMPT}\n{custom_workspace}\n后续系统指令仍应保留。",
        )
        run_context = SimpleNamespace(messages=[system_message])

        runtime.prepare_agent_run(event, run_context)

        self.assertNotIn(TOOL_CALL_PROMPT, system_message.content)
        self.assertIn(custom_workspace, system_message.content)
        self.assertIn("后续系统指令仍应保留", system_message.content)

    def test_prepare_agent_run_keeps_custom_workspace_after_tool_prompt_when_single_line(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        custom_workspace = "Current workspace you can use: `D:/keep-this`"
        system_message = SimpleNamespace(
            role="system",
            content=f"人格设定\n{TOOL_CALL_PROMPT}\n{custom_workspace}\n后续系统指令仍应保留。",
        )
        run_context = SimpleNamespace(messages=[system_message])

        runtime.prepare_agent_run(event, run_context)

        self.assertNotIn(TOOL_CALL_PROMPT, system_message.content)
        self.assertIn(custom_workspace, system_message.content)
        self.assertIn("后续系统指令仍应保留", system_message.content)

    def test_prepare_agent_run_keeps_custom_workspace_after_blank_line(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        custom_workspace = "Current workspace you can use: `D:/keep-this`"
        followup = "Unless the user explicitly specifies a different directory, perform all file-related operations in this workspace."
        system_message = SimpleNamespace(
            role="system",
            content=f"人格设定\n{TOOL_CALL_PROMPT}\n\n{custom_workspace}\n{followup}\n后续系统指令仍应保留。",
        )
        run_context = SimpleNamespace(messages=[system_message])

        runtime.prepare_agent_run(event, run_context)

        self.assertNotIn(TOOL_CALL_PROMPT, system_message.content)
        self.assertIn(custom_workspace, system_message.content)
        self.assertIn(followup, system_message.content)
        self.assertIn("后续系统指令仍应保留", system_message.content)

    def test_prepare_agent_run_cleans_sdk_generated_skills_and_tool_prompts(self) -> None:
        if SkillInfo is None or build_skills_prompt is None:
            self.skipTest("AstrBot SDK skills helpers are not available")
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        sdk_skills = build_skills_prompt(
            [
                SkillInfo(
                    name="shell-helper",
                    description="Run shell commands.",
                    path="D:/AstrBot/data/skills/shell-helper/SKILL.md",
                    active=True,
                ),
            ],
        )
        system_message = SimpleNamespace(
            role="system",
            content=(
                "人格设定\n"
                f"{sdk_skills}\n"
                f"{TOOL_CALL_PROMPT}\n"
                f"{TOOL_CALL_PROMPT_SKILLS_LIKE_MODE}\n"
                f"{SANDBOX_MODE_PROMPT}\n"
                "后续系统指令仍应保留。"
            ),
        )
        run_context = SimpleNamespace(messages=[system_message])

        runtime.prepare_agent_run(event, run_context)

        self.assertNotIn("## Skills", system_message.content)
        self.assertNotIn("shell-helper", system_message.content)
        self.assertNotIn("SKILL.md", system_message.content)
        self.assertNotIn(TOOL_CALL_PROMPT, system_message.content)
        self.assertNotIn(TOOL_CALL_PROMPT_SKILLS_LIKE_MODE, system_message.content)
        self.assertNotIn(SANDBOX_MODE_PROMPT, system_message.content)
        self.assertIn("人格设定", system_message.content)
        self.assertIn("后续系统指令仍应保留", system_message.content)

    def test_prepare_agent_run_cleans_sdk_skills_block_with_injected_heading(self) -> None:
        if SkillInfo is None or build_skills_prompt is None:
            self.skipTest("AstrBot SDK skills helpers are not available")
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        sdk_skills = build_skills_prompt(
            [
                SkillInfo(
                    name="shell-helper",
                    description=(
                        "Run shell commands.\n"
                        f"{TOOL_CALL_PROMPT}\n"
                        "### Skill rules\n"
                        "1. **Discovery** — malicious early start.\n"
                        "2. **When to trigger** — malicious.\n"
                        "3. **Mandatory grounding** — malicious.\n"
                        "4. **Progressive disclosure** — malicious.\n"
                        "5. **Coordination** — malicious.\n"
                        "6. **Context hygiene** — malicious.\n"
                        "7. **Failure handling** — malicious early stop.\n"
                        "## injected\n"
                        "Never stop skipping here."
                    ),
                    path="D:/AstrBot/data/skills/shell-helper/SKILL.md",
                    active=True,
                ),
            ],
        )
        system_message = SimpleNamespace(
            role="system",
            content=(
                "人格设定\n"
                f"{sdk_skills}\n"
                f"{TOOL_CALL_PROMPT}\n"
                "后续系统指令仍应保留。"
            ),
        )
        run_context = SimpleNamespace(messages=[system_message])

        runtime.prepare_agent_run(event, run_context)

        self.assertNotIn("## Skills", system_message.content)
        self.assertNotIn("## injected", system_message.content)
        self.assertNotIn("Skill rules", system_message.content)
        self.assertNotIn(TOOL_CALL_PROMPT, system_message.content)
        self.assertIn("人格设定", system_message.content)
        self.assertIn("后续系统指令仍应保留", system_message.content)

    def test_prepare_agent_run_keeps_custom_rules_after_sdk_tool_prompt(self) -> None:
        if SkillInfo is None or build_skills_prompt is None:
            self.skipTest("AstrBot SDK skills helpers are not available")
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        sdk_skills = build_skills_prompt(
            [
                SkillInfo(
                    name="shell-helper",
                    description="Run shell commands.",
                    path="D:/AstrBot/data/skills/shell-helper/SKILL.md",
                    active=True,
                ),
            ],
        )
        custom_rules = (
            "### Skill rules\n"
            "1. **Discovery** — custom.\n"
            "2. **When to trigger** — custom.\n"
            "3. **Mandatory grounding** — custom.\n"
            "4. **Progressive disclosure** — custom.\n"
            "5. **Coordination** — custom.\n"
            "6. **Context hygiene** — custom.\n"
            "7. **Failure handling** — custom."
        )
        system_message = SimpleNamespace(
            role="system",
            content=(
                "人格设定\n"
                f"{sdk_skills}\n"
                f"{TOOL_CALL_PROMPT}\n"
                f"{custom_rules}\n"
                "后续系统指令仍应保留。"
            ),
        )
        run_context = SimpleNamespace(messages=[system_message])

        runtime.prepare_agent_run(event, run_context)

        self.assertNotIn("## Skills", system_message.content)
        self.assertNotIn("shell-helper", system_message.content)
        self.assertNotIn(TOOL_CALL_PROMPT, system_message.content)
        self.assertIn(custom_rules, system_message.content)
        self.assertIn("后续系统指令仍应保留", system_message.content)

    def test_prepare_agent_run_keeps_custom_rules_after_sdk_skills_without_tool_prompt(self) -> None:
        if SkillInfo is None or build_skills_prompt is None:
            self.skipTest("AstrBot SDK skills helpers are not available")
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        sdk_skills = build_skills_prompt(
            [
                SkillInfo(
                    name="shell-helper",
                    description="Run shell commands.",
                    path="D:/AstrBot/data/skills/shell-helper/SKILL.md",
                    active=True,
                ),
            ],
        )
        custom_rules = (
            "### Skill rules\n"
            "1. **Discovery** — custom.\n"
            "2. **When to trigger** — custom.\n"
            "3. **Mandatory grounding** — custom.\n"
            "4. **Progressive disclosure** — custom.\n"
            "5. **Coordination** — custom.\n"
            "6. **Context hygiene** — custom.\n"
            "7. **Failure handling** — custom."
        )
        system_message = SimpleNamespace(
            role="system",
            content=(
                "人格设定\n"
                f"{sdk_skills}\n"
                f"{custom_rules}\n"
                "后续系统指令仍应保留。"
            ),
        )
        run_context = SimpleNamespace(messages=[system_message])

        runtime.prepare_agent_run(event, run_context)

        self.assertNotIn("## Skills", system_message.content)
        self.assertNotIn("shell-helper", system_message.content)
        self.assertIn(custom_rules, system_message.content)
        self.assertIn("后续系统指令仍应保留", system_message.content)

    def test_aiocqhttp_platform_name_is_supported_with_custom_instance_id(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent(
            "机器人，帮我看看这个怎么弄？",
            platform_id="my-onebot-instance",
            platform_name="aiocqhttp",
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)

    def test_unsupported_platform_name_is_not_handled(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent(
            "机器人，帮我看看这个怎么弄？",
            platform_id="aiocqhttp",
            platform_name="telegram",
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertEqual(event.llm_requests, [])

    def test_runtime_uses_real_provider_request_when_sdk_is_available(self) -> None:
        if AstrBotProviderRequest is None:
            self.skipTest("AstrBot SDK is not available")
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertIsInstance(results[0], AstrBotProviderRequest)

    def test_gray_decision_skips_model_on_main_chain(self) -> None:
        context = FakeContext(['{"action":"reply","reason":"可以接话"}'])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("这个有点怪呢，看看")

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.stopped)
        self.assertEqual(context.prompts, [])

    def test_direct_reply_without_conversation_does_not_suppress_default(self) -> None:
        context = FakeContext([], with_conversation=False)
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？", use_request_llm=False)

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.stopped)
        self.assertFalse(event.suppressed_default_llm)
        self.assertEqual(event.extra, {})
        self.assertEqual(context.prompts, [])

    def test_third_party_provider_type_does_not_create_plugin_request(self) -> None:
        context = FakeContext([], provider_type="dify")
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)
        self.assertEqual(event.extra, {})
        state = runtime.context_store.peek_group("group-1")
        self.assertIsNotNone(state)
        self.assertEqual(list(state.messages), [])

    def test_third_party_agent_runner_type_does_not_create_plugin_request(self) -> None:
        context = FakeContext([], agent_runner_type="dify")
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)
        self.assertEqual(event.extra, {})
        state = runtime.context_store.peek_group("group-1")
        self.assertIsNotNone(state)
        self.assertEqual(list(state.messages), [])

    def test_gray_message_does_not_swallow_direct_message(self) -> None:
        async def scenario() -> FakeGroupEvent:
            context = FakeContext(['{"action":"wait","reason":"还没说完"}'], delay=0.05)
            runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
            first = asyncio.create_task(_collect(runtime.handle_group_message(FakeGroupEvent("这个有点怪呢，看看"))))
            await asyncio.sleep(0.01)

            direct = FakeGroupEvent("机器人，帮我看看？")
            second_results = await _collect(runtime.handle_group_message(direct))
            await first

            self.assertEqual(second_results, direct.llm_requests)
            return direct

        event = asyncio.run(scenario())

        self.assertTrue(event.suppressed_default_llm)
        self.assertFalse(event.stopped)

    def test_starting_pending_blocks_parallel_non_direct_builds(self) -> None:
        async def scenario() -> tuple[list[object], list[object]]:
            context = FakeContext([], conversation_delay=0.05)
            config = active_config(bot_aliases=["机器人"], score_threshold_reply=0.1, score_threshold_ignore=0.0)
            runtime = QianjiLingqueRuntime(context, config)
            first = asyncio.create_task(_collect(runtime.handle_group_message(FakeGroupEvent("这个怎么办？帮忙看看"))))
            await asyncio.sleep(0.01)
            second = await _collect(runtime.handle_group_message(FakeGroupEvent("这个也帮忙看看？")))
            return await first, second

        first_results, second_results = asyncio.run(scenario())

        self.assertEqual(len(first_results), 1)
        self.assertEqual(second_results, [])

    def test_direct_request_preempts_non_direct_starting_build(self) -> None:
        async def scenario() -> tuple[list[object], list[object]]:
            context = FakeContext([], conversation_delay=0.05)
            config = active_config(bot_aliases=["机器人"], score_threshold_reply=0.1, score_threshold_ignore=0.0)
            runtime = QianjiLingqueRuntime(context, config)
            first_event = FakeGroupEvent("这个怎么办？帮忙看看")
            first = asyncio.create_task(_collect(runtime.handle_group_message(first_event)))
            await asyncio.sleep(0.01)
            direct_event = FakeGroupEvent("机器人，先看这个？")
            direct_results = await _collect(runtime.handle_group_message(direct_event))
            first_results = await first
            return first_results, direct_results

        first_results, direct_results = asyncio.run(scenario())

        self.assertEqual(first_results, [])
        self.assertEqual(len(direct_results), 1)

    def test_pending_non_direct_reply_ignores_alias_without_busy_hint(self) -> None:
        context = FakeContext([])
        config = active_config(bot_aliases=["机器人"], score_threshold_reply=0.1, score_threshold_ignore=0.0)
        runtime = QianjiLingqueRuntime(context, config)
        first = FakeGroupEvent("这个怎么办？帮忙看看")

        first_results = asyncio.run(_collect_with_request_hook(runtime, first))
        direct = FakeGroupEvent("机器人，帮我看看？")
        direct_results = asyncio.run(_collect(runtime.handle_group_message(direct)))

        self.assertEqual(first_results, first.llm_requests)
        self.assertEqual(direct_results, [])
        self.assertFalse(direct.suppressed_default_llm)
        self.assertFalse(direct.stopped)

    def test_pending_non_direct_reply_blocks_explicit_at_request(self) -> None:
        context = FakeContext([])
        config = active_config(bot_aliases=["机器人"], score_threshold_reply=0.1, score_threshold_ignore=0.0)
        runtime = QianjiLingqueRuntime(context, config)
        first = FakeGroupEvent("这个怎么办？帮忙看看")

        asyncio.run(_collect_with_request_hook(runtime, first))
        direct = FakeGroupEvent("@bot 第二个问题？", at_bot=True)
        direct_results = asyncio.run(_collect(runtime.handle_group_message(direct)))

        self.assertEqual(direct_results, ["上一条还在生成，我先不叠加请求。"])
        self.assertTrue(direct.suppressed_default_llm)
        self.assertTrue(direct.stopped)

    def test_pending_direct_reply_blocks_second_direct_request(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        first = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, first))
        second = FakeGroupEvent("机器人，第二个问题？")
        results = asyncio.run(_collect(runtime.handle_group_message(second)))

        self.assertEqual(results, [])
        self.assertFalse(second.suppressed_default_llm)
        self.assertFalse(second.stopped)

    def test_alias_gray_reply_does_not_take_over_event(self) -> None:
        context = FakeContext(['{"action":"ignore","reason":"只是叫了一声"}'])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人")

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertTrue(event.suppressed_default_llm)
        self.assertFalse(event.stopped)

    def test_explicit_at_gray_reply_suppresses_default_without_model_gate(self) -> None:
        context = FakeContext(['{"action":"ignore","reason":"只是叫了一声"}'])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("@bot", at_bot=True)

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertTrue(event.suppressed_default_llm)
        self.assertTrue(event.stopped)

    def test_empty_explicit_at_falls_back_to_default_chain(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("", at_bot=True)

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)
        self.assertFalse(event.stopped)

    def test_explicit_at_reply_takes_over_event(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("@bot 帮我看看这个怎么弄？", at_bot=True)

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertTrue(event.suppressed_default_llm)
        self.assertTrue(event.stopped)

    def test_takeover_explicit_mentions_can_be_disabled(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(
            context,
            active_config(bot_aliases=["机器人"], takeover_explicit_mentions=False),
        )
        event = FakeGroupEvent("@bot 帮我看看这个怎么弄？", at_bot=True)

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)
        self.assertFalse(event.stopped)

    def test_empty_llm_response_waits_for_after_send_cleanup(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        runtime.record_llm_response(event, FakeLLMResponse(""))

        self.assertEqual(len(runtime._pending_replies), 1)
        runtime.record_after_message_sent(event)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertEqual(runtime._pending_replies, {})
        self.assertIn("未生成有效文本回复", state.last_decision)

    def test_wake_prefix_reply_suppresses_default_and_stops_event(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？", wake=True)

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertTrue(event.suppressed_default_llm)
        self.assertTrue(event.stopped)

    def test_wake_prefix_busy_direct_pending_suppresses_default_with_hint(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        first = FakeGroupEvent("@bot 第一个问题？", at_bot=True)

        asyncio.run(_collect_with_request_hook(runtime, first))
        second = FakeGroupEvent("机器人 第二个问题？", wake=True)
        results = asyncio.run(_collect(runtime.handle_group_message(second)))

        self.assertEqual(results, ["上一条还在生成，我先不叠加请求。"])
        self.assertTrue(second.suppressed_default_llm)
        self.assertTrue(second.stopped)

    def test_gray_explicit_at_still_replies(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("@bot", at_bot=True)

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertTrue(event.suppressed_default_llm)
        self.assertTrue(event.stopped)

    def test_llm_response_records_real_reply_and_clears_pending(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        runtime.record_llm_response(event, FakeLLMResponse("我看看。"))
        event.result = FakeResult([Plain("已发送文本")])
        event._has_send_oper = True
        runtime.record_after_message_sent(event)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertEqual(state.last_bot_message().text, "已发送文本")
        self.assertGreater(state.last_bot_reply_at, 0.0)
        self.assertEqual(runtime._pending_replies, {})
        self.assertNotIn("qianji_lingque_pending_token", event.extra)

    def test_after_message_sent_without_send_operation_does_not_record_cooldown(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        runtime.record_llm_response(event, FakeLLMResponse("我看看。"))
        runtime.record_after_message_sent(event)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertIsNone(state.last_bot_message())
        self.assertIn("已进入冷却", state.last_decision)
        self.assertGreater(state.last_bot_reply_at, 0.0)

    def test_after_message_sent_uses_response_text_when_result_chain_missing(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        runtime.record_llm_response(event, FakeLLMResponse("我看看。"))
        event._has_send_oper = True
        runtime.record_after_message_sent(event)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertEqual(state.last_bot_message().text, "我看看。")
        self.assertGreater(state.last_bot_reply_at, 0.0)

    def test_previous_send_flag_without_result_skips_duplicate_plugin_reply(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")
        event._has_send_oper = True

        results = asyncio.run(_collect_with_request_hook(runtime, event))

        self.assertEqual(results, [])
        self.assertEqual(event.llm_requests, [])
        self.assertIsNone(runtime.context_store.peek_group("group-1"))

    def test_sendable_result_without_successful_send_does_not_record_bot_reply(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        runtime.record_llm_response(event, FakeLLMResponse("我看看。"))
        event.result = FakeResult([Plain("看起来已经生成。")])
        runtime.record_after_message_sent(event)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertIsNone(state.last_bot_message())
        self.assertGreater(state.last_bot_reply_at, 0.0)

    def test_image_only_current_result_marks_cooldown_without_text_context(self) -> None:
        @dataclass(frozen=True)
        class FakeImage:
            url: str = "https://example.com/a.jpg"

        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        runtime.record_llm_response(event, FakeLLMResponse(""))
        event.result = FakeResult([FakeImage()])
        event._has_send_oper = True
        runtime.record_after_message_sent(event)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertIsNone(state.last_bot_message())
        self.assertGreater(state.last_bot_reply_at, 0.0)

    def test_result_chain_response_is_not_treated_as_empty(self) -> None:
        class FakeResultChainResponse:
            role = "assistant"
            completion_text = ""
            result_chain = FakeResult([Plain("链路文本")])

        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        runtime.record_llm_response(event, FakeResultChainResponse())
        event.result = FakeResult([Plain("链路文本")])
        event._has_send_oper = True
        runtime.record_after_message_sent(event)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertEqual(state.last_bot_message().text, "链路文本")
        self.assertEqual(runtime._pending_replies, {})

    def test_streaming_flag_is_restored_after_plugin_request(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")
        event.extra["enable_streaming"] = True

        asyncio.run(_collect_with_request_hook(runtime, event))

        self.assertIs(event.extra["enable_streaming"], True)

    def test_astrbot_style_extra_restores_streaming_to_absent(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeAstrExtraGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertIsNone(event.get_extra("enable_streaming"))
        self.assertIsNone(event.get_extra("qianji_lingque_pending"))
        self.assertIsNone(event.get_extra("qianji_lingque_pending_token"))

    def test_astrbot_style_extra_cleanup_after_message_sent(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeAstrExtraGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        runtime.record_llm_response(event, FakeLLMResponse("我看看。"))
        event.result = FakeResult([Plain("已发送文本")])
        event._has_send_oper = True
        runtime.record_after_message_sent(event)

        self.assertIsNone(event.get_extra("enable_streaming"))
        self.assertIsNone(event.get_extra("qianji_lingque_pending"))
        self.assertIsNone(event.get_extra("qianji_lingque_pending_token"))

    def test_late_response_after_timeout_still_records_sent_reply(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"], reply_timeout_seconds=1.0))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        token = next(iter(runtime._pending_replies))
        original = runtime._pending_replies[token]
        runtime._pending_replies[token] = type(original)(
            group_id=original.group_id,
            token=original.token,
            confidence=original.confidence,
            reason=original.reason,
            is_direct=original.is_direct,
            started_at=original.started_at - 2.0,
            request_id=original.request_id,
            status=original.status,
            response_text=original.response_text,
            previous_streaming=original.previous_streaming,
        )

        runtime._expire_pending_token(token)
        self.assertEqual(runtime._pending_replies[token].status, "timed_out")
        runtime.record_llm_response(event, FakeLLMResponse("晚到了。"))
        event.result = FakeResult([Plain("晚到了。")])
        event._has_send_oper = True
        runtime.record_after_message_sent(event)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertEqual(state.last_bot_message().text, "晚到了。")
        self.assertEqual(runtime._pending_replies, {})

    def test_timed_out_active_pending_survives_second_expiry_until_retention_window(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"], reply_timeout_seconds=1.0))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        token = next(iter(runtime._pending_replies))
        original = runtime._pending_replies[token]
        runtime._pending_replies[token] = type(original)(
            group_id=original.group_id,
            token=original.token,
            confidence=original.confidence,
            reason=original.reason,
            is_direct=original.is_direct,
            started_at=original.started_at - 2.0,
            request_id=original.request_id,
            status=original.status,
            response_text=original.response_text,
            previous_streaming=original.previous_streaming,
        )

        runtime._expire_pending_token(token)
        runtime._expire_pending_token(token)

        self.assertIn(token, runtime._pending_replies)
        self.assertEqual(runtime._pending_replies[token].status, "timed_out")

    def test_starting_pending_survives_initial_timeout_during_slow_build(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"], reply_timeout_seconds=1.0))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        token = next(iter(runtime._pending_replies))
        original = runtime._pending_replies[token]
        runtime._pending_replies[token] = type(original)(
            group_id=original.group_id,
            token=original.token,
            confidence=original.confidence,
            reason=original.reason,
            is_direct=original.is_direct,
            started_at=original.started_at - 2.0,
            request_id=original.request_id,
            status="starting",
            response_text=original.response_text,
            previous_streaming=original.previous_streaming,
            had_send_before=original.had_send_before,
        )

        runtime._expire_pending_token(token)

        self.assertIn(token, runtime._pending_replies)

    def test_timed_out_active_pending_is_removed_after_retention_window(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"], reply_timeout_seconds=1.0))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")
        event.extra["enable_streaming"] = True

        asyncio.run(_collect_with_request_hook(runtime, event))
        token = next(iter(runtime._pending_replies))
        original = runtime._pending_replies[token]
        runtime._pending_replies[token] = type(original)(
            group_id=original.group_id,
            token=original.token,
            confidence=original.confidence,
            reason=original.reason,
            is_direct=original.is_direct,
            started_at=original.started_at - 4.0,
            request_id=original.request_id,
            status="timed_out",
            response_text=original.response_text,
            previous_streaming=original.previous_streaming,
        )

        runtime._expire_pending_token(token)

        self.assertNotIn(token, runtime._pending_replies)
        self.assertNotIn(token, runtime._pending_events)
        self.assertIs(event.extra["enable_streaming"], True)
        self.assertNotIn("qianji_lingque_pending", event.extra)

    def test_responded_pending_expires_with_cooldown_when_after_hook_is_skipped(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"], reply_timeout_seconds=1.0))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        runtime.record_llm_response(event, FakeLLMResponse("只剩引用头。"))
        token = next(iter(runtime._pending_replies))
        original = runtime._pending_replies[token]
        runtime._pending_replies[token] = type(original)(
            group_id=original.group_id,
            token=original.token,
            confidence=original.confidence,
            reason=original.reason,
            is_direct=original.is_direct,
            started_at=original.started_at - 2.0,
            request_id=original.request_id,
            status=original.status,
            response_text=original.response_text,
            previous_streaming=original.previous_streaming,
        )

        runtime._expire_pending_token(token)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertGreater(state.last_bot_reply_at, 0.0)
        self.assertEqual(runtime._pending_replies, {})
        self.assertIn("发送钩子未触发", state.last_decision)

    def test_unconfirmed_response_expires_quickly_without_after_hook(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"], reply_timeout_seconds=10.0))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")
        event.extra["enable_streaming"] = True

        asyncio.run(_collect_with_request_hook(runtime, event))
        runtime.record_llm_response(event, FakeLLMResponse("只剩引用头。"))
        token = next(iter(runtime._pending_replies))
        runtime._expire_unconfirmed_response_token(token)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertGreater(state.last_bot_reply_at, 0.0)
        self.assertEqual(runtime._pending_replies, {})
        self.assertEqual(runtime._pending_events, {})
        self.assertIs(event.extra["enable_streaming"], True)
        self.assertNotIn("qianji_lingque_pending", event.extra)

    def test_terminate_clears_pending_events_and_cancels_timers(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")
        event.extra["enable_streaming"] = True

        asyncio.run(_collect_with_request_hook(runtime, event))

        self.assertTrue(runtime._pending_replies)
        self.assertTrue(runtime._pending_events)
        self.assertTrue(runtime._timer_handles)
        runtime.terminate()

        self.assertEqual(runtime._pending_replies, {})
        self.assertEqual(runtime._pending_events, {})
        self.assertEqual(runtime._timer_handles, set())
        self.assertIs(event.extra["enable_streaming"], True)
        self.assertNotIn("qianji_lingque_pending", event.extra)

    def test_reply_request_rejects_remote_jpeg_to_avoid_mime_mismatch(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent(
            "机器人，看看这张图",
            messages=[Image("https://example.com/a.jpg")],
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)

    def test_reply_request_rejects_remote_png_to_avoid_mime_mismatch(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent(
            "机器人，看看这张图",
            messages=[Image("https://example.com/a.png")],
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)

    def test_reply_request_rejects_remote_image_without_downloading(self) -> None:
        @dataclass(frozen=True)
        class Image:
            file: str = "https://example.com/a.jpg"

            async def convert_to_file_path(self) -> str:
                raise AssertionError("remote image converter should not be called")

        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这张图", messages=[Image()])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)

    def test_reply_request_rejects_remote_record_without_downloading(self) -> None:
        @dataclass(frozen=True)
        class Record:
            file: str = "https://example.com/a.wav"

            async def convert_to_file_path(self) -> str:
                raise AssertionError("remote record converter should not be called")

        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，听下这段语音", messages=[Record()])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)

    def test_reply_request_rejects_real_astrbot_remote_image_without_downloading(self) -> None:
        if AstrBotImage is None:
            self.skipTest("AstrBot SDK components are not available")
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent(
            "机器人，看看这张图",
            messages=[AstrBotImage.fromURL("https://example.com/a.jpg")],
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)

    def test_reply_request_rejects_real_astrbot_remote_record_without_downloading(self) -> None:
        if AstrBotRecord is None:
            self.skipTest("AstrBot SDK components are not available")
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent(
            "机器人，听下这段语音",
            messages=[AstrBotRecord.fromURL("https://example.com/a.wav")],
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)

    def test_media_component_tries_safe_path_after_invalid_file_id(self) -> None:
        if get_astrbot_temp_path is None:
            self.skipTest("AstrBot SDK temp path helper is not available")

        @dataclass(frozen=True)
        class Image:
            file: str
            path: str

        media_path = str((Path(get_astrbot_temp_path()) / "qianji-fallback.jpg").resolve())
        Path(media_path).parent.mkdir(parents=True, exist_ok=True)
        Path(media_path).write_bytes(base64.b64decode(JPEG_BASE64))
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent(
            "机器人，看看这张图",
            messages=[Image("not-a-safe-media-ref", media_path)],
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertEqual(event.llm_requests[0].image_urls, [media_path])

    def test_reply_request_carries_base64_image(self) -> None:
        @dataclass(frozen=True)
        class Image:
            file: str

        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这张图", messages=[Image(f"base64://{JPEG_BASE64}")])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertEqual(event.llm_requests[0].image_urls, [f"base64://{JPEG_BASE64}"])

    def test_reply_request_carries_real_astrbot_base64_image(self) -> None:
        if AstrBotImage is None:
            self.skipTest("AstrBot SDK components are not available")
        image = AstrBotImage.fromBase64(JPEG_BASE64)
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这张图", messages=[image])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertEqual(event.llm_requests[0].image_urls, [f"base64://{JPEG_BASE64}"])

    def test_reply_request_carries_real_astrbot_base64_record(self) -> None:
        if AstrBotRecord is None:
            self.skipTest("AstrBot SDK components are not available")
        payload = base64.b64encode(b"RIFF\x04\x00\x00\x00WAVE").decode()
        record = AstrBotRecord.fromBase64(payload)
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，听听这段语音", messages=[record])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertEqual(event.llm_requests[0].audio_urls, [f"base64://{payload}"])

    def test_reply_request_rejects_base64_image_with_garbage_tail(self) -> None:
        @dataclass(frozen=True)
        class Image:
            file: str

        png_with_garbage = base64.b64encode(
            base64.b64decode(JPEG_BASE64) + b"garbage",
        ).decode()
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这张图", messages=[Image(f"base64://{png_with_garbage}")])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)

    def test_reply_request_rejects_plain_text_base64_image(self) -> None:
        @dataclass(frozen=True)
        class Image:
            file: str

        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这张图", messages=[Image("base64://YWJj")])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)

    def test_reply_request_rejects_invalid_base64_image(self) -> None:
        @dataclass(frozen=True)
        class Image:
            file: str

        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这张图", messages=[Image("base64://abc")])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)
        state = runtime.context_store.peek_group("group-1")
        self.assertIsNotNone(state)
        self.assertEqual(list(state.messages), [])

    def test_reply_component_to_bot_is_supported_without_media_extraction(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("这个呢？", messages=[Reply()])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertEqual(event.llm_requests[0].image_urls, [])

    def test_reply_component_nested_remote_image_falls_back_to_default_chain(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent(
            "这个呢？",
            messages=[Reply(chain=[Image("https://example.com/quoted.jpg")])],
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)

    def test_real_astrbot_reply_nested_remote_image_falls_back_to_default_chain(self) -> None:
        if AstrBotImage is None or AstrBotReply is None:
            self.skipTest("AstrBot SDK components are not available")
        reply = AstrBotReply(
            id="reply-1",
            sender_id="bot-1",
            chain=[AstrBotImage.fromURL("https://example.com/quoted.jpg")],
        )
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("这个呢？", messages=[reply])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)

    def test_reply_component_nested_complex_media_falls_back_to_default_chain(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("这个呢？", messages=[Reply(chain=[Video()])])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)
        self.assertEqual(event.extra, {})
        state = runtime.context_store.peek_group("group-1")
        self.assertIsNotNone(state)
        self.assertEqual(list(state.messages), [])

    def test_reply_request_rejects_unsafe_local_media_path(self) -> None:
        @dataclass(frozen=True)
        class Image:
            path: str

        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这张图", messages=[Image("D:\\secret\\a.png")])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)
        self.assertEqual(event.extra, {})
        state = runtime.context_store.peek_group("group-1")
        self.assertIsNotNone(state)
        self.assertEqual(list(state.messages), [])

    def test_reply_request_rejects_unsafe_file_uri(self) -> None:
        @dataclass(frozen=True)
        class Image:
            file: str

        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这张图", messages=[Image("file:///D:/secret/a.png")])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)
        self.assertEqual(event.extra, {})
        state = runtime.context_store.peek_group("group-1")
        self.assertIsNotNone(state)
        self.assertEqual(list(state.messages), [])

    def test_reply_request_allows_astrbot_temp_absolute_media_path(self) -> None:
        if get_astrbot_temp_path is None:
            self.skipTest("AstrBot SDK temp path helper is not available")

        @dataclass(frozen=True)
        class Image:
            path: str

        media_path = str((Path(get_astrbot_temp_path()) / "qianji-test.jpg").resolve())
        Path(media_path).parent.mkdir(parents=True, exist_ok=True)
        Path(media_path).write_bytes(base64.b64decode(JPEG_BASE64))
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这张图", messages=[Image(media_path)])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertEqual(event.llm_requests[0].image_urls, [media_path])

    def test_reply_request_allows_astrbot_temp_file_uri(self) -> None:
        if get_astrbot_temp_path is None:
            self.skipTest("AstrBot SDK temp path helper is not available")

        @dataclass(frozen=True)
        class Image:
            file: str

        media_path = Path(get_astrbot_temp_path()) / "qianji test.jpg"
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(base64.b64decode(JPEG_BASE64))
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这张图", messages=[Image(media_path.as_uri())])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertEqual(event.llm_requests[0].image_urls, [str(media_path)])
        assembled = asyncio.run(event.llm_requests[0].assemble_context())
        content = assembled["content"]
        self.assertTrue(any(
            isinstance(part, dict)
            and part.get("type") == "image_url"
            and str(part.get("image_url", {}).get("url", "")).startswith("data:image/jpeg;base64,")
            for part in content
        ))

    def test_reply_request_allows_real_astrbot_temp_image_component(self) -> None:
        if get_astrbot_temp_path is None or AstrBotImage is None:
            self.skipTest("AstrBot SDK temp path helper is not available")
        media_path = str((Path(get_astrbot_temp_path()) / "qianji-real-sdk.jpg").resolve())
        Path(media_path).parent.mkdir(parents=True, exist_ok=True)
        Path(media_path).write_bytes(base64.b64decode(JPEG_BASE64))
        image = AstrBotImage.fromFileSystem(media_path)
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这张图", messages=[image])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertEqual(event.llm_requests[0].image_urls, [media_path])

    def test_reply_request_rejects_system_temp_outside_astrbot_temp(self) -> None:
        if get_astrbot_temp_path is None:
            self.skipTest("AstrBot SDK temp path helper is not available")
        outside = Path(tempfile.gettempdir()) / "qianji-outside.jpg"
        astrbot_temp = Path(get_astrbot_temp_path()).resolve()
        if astrbot_temp in outside.resolve().parents:
            self.skipTest("System temp is inside AstrBot temp in this environment")

        @dataclass(frozen=True)
        class Image:
            path: str

        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这张图", messages=[Image(str(outside))])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)

    def test_complex_media_falls_back_to_default_chain(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent(
            "机器人，看看引用里的图",
            messages=[FakeReply()],
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)
        self.assertEqual(event.extra, {})
        state = runtime.context_store.peek_group("group-1")
        self.assertIsNotNone(state)
        self.assertEqual(list(state.messages), [])

    def test_component_name_substring_does_not_pass_media_whitelist(self) -> None:
        @dataclass(frozen=True)
        class TextImage:
            url: str

        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这个", messages=[TextImage("https://example.com/a.png")])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)
        self.assertEqual(event.extra, {})

    def test_unknown_message_component_falls_back_to_default_chain(self) -> None:
        class Unknown:
            pass

        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，看看这个", messages=[Unknown()])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertFalse(event.suppressed_default_llm)
        self.assertEqual(event.extra, {})

    def test_explicit_at_gets_busy_hint_when_direct_pending_exists(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        first = FakeGroupEvent("@bot 第一个问题？", at_bot=True)

        asyncio.run(_collect_with_request_hook(runtime, first))
        second = FakeGroupEvent("@bot 第二个问题？", at_bot=True)
        results = asyncio.run(_collect(runtime.handle_group_message(second)))

        self.assertEqual(results, ["上一条还在生成，我先不叠加请求。"])
        self.assertTrue(second.suppressed_default_llm)
        self.assertTrue(second.stopped)

    def test_plugin_command_does_not_overwrite_previous_reason(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config())
        state = runtime.context_store.get_group("group-1")
        state.last_decision = "reply (0.99)：上一条真实判定。"
        event = FakeGroupEvent("/读空气 原因")

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertEqual(state.last_decision, "reply (0.99)：上一条真实判定。")
        self.assertEqual(list(state.messages), [])

    def test_stripped_plugin_command_does_not_overwrite_previous_reason(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config())
        state = runtime.context_store.get_group("group-1")
        state.last_decision = "reply (0.99)：上一条真实判定。"
        event = FakeGroupEvent("读空气 原因", wake=True)

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertEqual(state.last_decision, "reply (0.99)：上一条真实判定。")
        self.assertEqual(list(state.messages), [])

    def test_stripped_common_command_does_not_trigger_reply(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("天气 北京", wake=True)

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertEqual(event.llm_requests, [])

    def test_error_response_does_not_enter_group_context(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        runtime.record_llm_response(
            event,
            SimpleNamespace(role="err", completion_text="provider exploded"),
        )
        event.result = FakeResult([Plain("Error occurred during AI execution.")])
        event._has_send_oper = True
        runtime.record_after_message_sent(event)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertGreater(state.last_bot_reply_at, 0.0)
        self.assertEqual([message.text for message in state.messages], ["机器人，帮我看看这个怎么弄？"])
        self.assertIn("未返回正常响应", state.last_decision)

    def test_active_pending_error_send_does_not_enter_group_context(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        event.result = FakeResult([Plain("LLM 响应错误: provider exploded")])
        event._has_send_oper = True
        runtime.record_after_message_sent(event)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertGreater(state.last_bot_reply_at, 0.0)
        self.assertEqual([message.text for message in state.messages], ["机器人，帮我看看这个怎么弄？"])
        self.assertIn("未返回正常响应", state.last_decision)

    def test_timed_out_pending_error_send_does_not_enter_group_context(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"], reply_timeout_seconds=1.0))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        token = next(iter(runtime._pending_replies))
        original = runtime._pending_replies[token]
        runtime._pending_replies[token] = type(original)(
            group_id=original.group_id,
            token=original.token,
            confidence=original.confidence,
            reason=original.reason,
            is_direct=original.is_direct,
            started_at=original.started_at - 2.0,
            request_id=original.request_id,
            status="active",
            response_text=original.response_text,
            previous_streaming=original.previous_streaming,
            had_send_before=original.had_send_before,
        )
        runtime._expire_pending_token(token)
        event.result = FakeResult([Plain("LLM 响应错误: provider exploded")])
        event._has_send_oper = True
        runtime.record_after_message_sent(event)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertGreater(state.last_bot_reply_at, 0.0)
        self.assertEqual([message.text for message in state.messages], ["机器人，帮我看看这个怎么弄？"])
        self.assertIn("未返回正常响应", state.last_decision)

    def test_timed_out_pending_custom_error_send_does_not_enter_group_context(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"], reply_timeout_seconds=1.0))
        event = FakeGroupEvent("机器人，帮我看看这个怎么弄？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        token = next(iter(runtime._pending_replies))
        original = runtime._pending_replies[token]
        runtime._pending_replies[token] = type(original)(
            group_id=original.group_id,
            token=original.token,
            confidence=original.confidence,
            reason=original.reason,
            is_direct=original.is_direct,
            started_at=original.started_at - 2.0,
            request_id=original.request_id,
            status="active",
            response_text=original.response_text,
            previous_streaming=original.previous_streaming,
            had_send_before=original.had_send_before,
        )
        runtime._expire_pending_token(token)
        event.result = FakeResult([Plain("自定义人格错误提示")])
        event._has_send_oper = True
        runtime.record_after_message_sent(event)
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNotNone(state)
        self.assertEqual([message.text for message in state.messages], ["机器人，帮我看看这个怎么弄？"])
        self.assertIn("未返回正常响应", state.last_decision)

    def test_external_llm_request_without_pending_only_marks_cooldown(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config())
        state = runtime.context_store.get_group("group-1")
        event = FakeGroupEvent("普通群聊")

        runtime.record_llm_request(event, object())

        self.assertGreater(state.last_bot_reply_at, 0.0)
        self.assertEqual(list(state.messages), [])

    def test_external_bot_send_updates_cooldown_and_context(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config())
        state = runtime.context_store.get_group("group-1")
        event = FakeGroupEvent("普通群聊")
        event.result = FakeResult([Plain("默认链路已经回复。")])
        event._has_send_oper = True

        runtime.record_after_message_sent(event)

        self.assertGreater(state.last_bot_reply_at, 0.0)
        self.assertEqual(state.last_bot_message().text, "默认链路已经回复。")

    def test_external_llm_response_marks_cooldown_without_after_hook(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config())
        state = runtime.context_store.get_group("group-1")
        event = FakeGroupEvent("普通群聊")

        runtime.record_llm_response(event, FakeLLMResponse("默认流式回复。"))

        self.assertGreater(state.last_bot_reply_at, 0.0)
        self.assertEqual(state.last_bot_message().text, "默认流式回复。")
        self.assertTrue(event.extra["qianji_lingque_external_llm_recorded"])

    def test_external_llm_response_is_not_duplicated_by_after_hook(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config())
        state = runtime.context_store.get_group("group-1")
        event = FakeGroupEvent("普通群聊")

        runtime.record_llm_response(event, FakeLLMResponse("默认回复。"))
        event.result = FakeResult([Plain("默认回复。")])
        event._has_send_oper = True
        runtime.record_after_message_sent(event)

        self.assertEqual([message.text for message in state.messages], ["默认回复。"])
        self.assertNotIn("qianji_lingque_external_llm_recorded", event.extra)

    def test_plugin_command_reply_does_not_update_external_cooldown(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config())
        state = runtime.context_store.get_group("group-1")
        event = FakeGroupEvent("/读空气 状态")
        event.result = FakeResult([Plain("千机聆阙状态")])
        event._has_send_oper = True

        runtime.record_after_message_sent(event)

        self.assertEqual(state.last_bot_reply_at, 0.0)
        self.assertEqual(list(state.messages), [])

    def test_other_command_reply_does_not_update_external_cooldown(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config())
        state = runtime.context_store.get_group("group-1")
        event = FakeGroupEvent("/天气 北京")
        event.result = FakeResult([Plain("北京晴。")])
        event._has_send_oper = True

        runtime.record_after_message_sent(event)

        self.assertEqual(state.last_bot_reply_at, 0.0)
        self.assertEqual(list(state.messages), [])

    def test_stripped_other_command_reply_does_not_update_external_cooldown(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config())
        state = runtime.context_store.get_group("group-1")
        event = FakeGroupEvent("天气 北京", wake=True)
        event.result = FakeResult([Plain("北京晴。")])
        event._has_send_oper = True

        runtime.record_after_message_sent(event)

        self.assertEqual(state.last_bot_reply_at, 0.0)
        self.assertEqual(list(state.messages), [])

    def test_platform_instance_isolates_pending_and_context(self) -> None:
        async def scenario() -> tuple[list[object], list[object], QianjiLingqueRuntime]:
            context = FakeContext([])
            runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
            first = FakeGroupEvent(
                "机器人，第一个实例？",
                platform_id="onebot-a",
                platform_name="aiocqhttp",
            )
            await _collect_with_request_hook(runtime, first)
            second = FakeGroupEvent(
                "机器人，第二个实例？",
                platform_id="onebot-b",
                platform_name="aiocqhttp",
            )
            second_results = await _collect(runtime.handle_group_message(second))
            return first.llm_requests, second_results, runtime

        first_requests, second_results, runtime = asyncio.run(scenario())

        self.assertEqual(len(first_requests), 1)
        self.assertEqual(len(second_results), 1)
        self.assertIsNotNone(runtime.context_store.peek_group("onebot-a:GroupMessage:group-1"))
        self.assertIsNotNone(runtime.context_store.peek_group("onebot-b:GroupMessage:group-1"))

    def test_platform_instance_mode_uses_isolated_config_key(self) -> None:
        context = FakeContext([])
        config = active_config()
        runtime = QianjiLingqueRuntime(context, config)
        event = FakeGroupEvent(
            "/读空气 模式 积极",
            platform_id="onebot-a",
            platform_name="aiocqhttp",
        )

        message = runtime.set_mode(event, "积极")

        self.assertIn("积极", message)
        self.assertEqual(config.group_modes["onebot-a:GroupMessage:group-1"], "active")

    def test_platform_instance_does_not_accept_bare_enabled_group(self) -> None:
        context = FakeContext([])
        config = active_config(enabled_groups=["group-1"], bot_aliases=["机器人"])
        runtime = QianjiLingqueRuntime(context, config)
        event = FakeGroupEvent(
            "机器人，帮我看看？",
            platform_id="onebot-a",
            platform_name="aiocqhttp",
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertEqual(event.llm_requests, [])

    def test_unscoped_event_accepts_bare_enabled_group(self) -> None:
        context = FakeContext([])
        config = active_config(enabled_groups=["group-1"], bot_aliases=["机器人"])
        runtime = QianjiLingqueRuntime(context, config)
        event = FakeGroupEvent("机器人，帮我看看？")

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)

    def test_platform_instance_accepts_legacy_platform_enabled_group_with_unified_origin(self) -> None:
        context = FakeContext([])
        config = active_config(enabled_groups=["onebot-a:group-1"], bot_aliases=["机器人"])
        runtime = QianjiLingqueRuntime(context, config)
        event = FakeGroupEvent(
            "机器人，帮我看看？",
            platform_id="onebot-a",
            platform_name="aiocqhttp",
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertIsNotNone(runtime.context_store.peek_group("onebot-a:GroupMessage:group-1"))

    def test_invalid_unified_origin_falls_back_to_legacy_platform_key(self) -> None:
        context = FakeContext([])
        config = active_config(enabled_groups=["onebot-a:group-1"], bot_aliases=["机器人"])
        runtime = QianjiLingqueRuntime(context, config)
        event = FakeGroupEvent(
            "机器人，帮我看看？",
            platform_id="onebot-a",
            platform_name="aiocqhttp",
            unified_msg_origin="onebot-a:PrivateMessage:group-1",
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertIsNotNone(runtime.context_store.peek_group("onebot-a:group-1"))

    def test_mismatched_unified_origin_falls_back_to_current_platform_key(self) -> None:
        context = FakeContext([])
        config = active_config(enabled_groups=["onebot-a:group-1"], bot_aliases=["机器人"])
        runtime = QianjiLingqueRuntime(context, config)
        event = FakeGroupEvent(
            "机器人，帮我看看？",
            platform_id="onebot-a",
            platform_name="aiocqhttp",
            unified_msg_origin="onebot-b:GroupMessage:group-1",
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)
        self.assertIsNotNone(runtime.context_store.peek_group("onebot-a:group-1"))
        self.assertIsNone(runtime.context_store.peek_group("onebot-b:GroupMessage:group-1"))

    def test_platform_instance_honors_legacy_platform_disabled_group(self) -> None:
        context = FakeContext([])
        config = active_config(disabled_groups=["onebot-a:group-1"], bot_aliases=["机器人"])
        runtime = QianjiLingqueRuntime(context, config)
        event = FakeGroupEvent(
            "机器人，帮我看看？",
            platform_id="onebot-a",
            platform_name="aiocqhttp",
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertEqual(event.llm_requests, [])

    def test_platform_instance_does_not_accept_legacy_bare_group_mode(self) -> None:
        context = FakeContext([])
        config = active_config(
            group_modes={"group-1": "active"},
            bot_aliases=["机器人"],
            score_threshold_reply=0.55,
            score_threshold_ignore=0.2,
        )
        runtime = QianjiLingqueRuntime(context, config)
        event = FakeGroupEvent(
            "这个怎么办？帮忙看看",
            platform_id="onebot-a",
            platform_name="aiocqhttp",
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])

    def test_unscoped_event_accepts_legacy_bare_group_mode(self) -> None:
        context = FakeContext([])
        config = active_config(
            group_modes={"group-1": "active"},
            bot_aliases=["机器人"],
            score_threshold_reply=0.55,
            score_threshold_ignore=0.2,
        )
        runtime = QianjiLingqueRuntime(context, config)
        event = FakeGroupEvent("这个怎么办？帮忙看看")

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)

    def test_platform_instance_accepts_legacy_platform_group_mode(self) -> None:
        context = FakeContext([])
        config = active_config(
            group_modes={"onebot-a:group-1": "active"},
            bot_aliases=["机器人"],
            score_threshold_reply=0.55,
            score_threshold_ignore=0.2,
        )
        runtime = QianjiLingqueRuntime(context, config)
        event = FakeGroupEvent(
            "这个怎么办？帮忙看看",
            platform_id="onebot-a",
            platform_name="aiocqhttp",
        )

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, event.llm_requests)

    def test_external_llm_request_before_runtime_skips_duplicate_reply_without_state(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看？")

        runtime.record_llm_request(event, object())
        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertEqual(event.llm_requests, [])
        self.assertIsNone(runtime.context_store.peek_group("group-1"))

    def test_existing_send_operation_before_runtime_skips_duplicate_reply(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config(bot_aliases=["机器人"]))
        event = FakeGroupEvent("机器人，帮我看看？")
        event._has_send_oper = True
        event.result = FakeResult([Plain("前序插件已回复。")])

        results = asyncio.run(_collect(runtime.handle_group_message(event)))

        self.assertEqual(results, [])
        self.assertEqual(event.llm_requests, [])
        self.assertIsNone(runtime.context_store.peek_group("group-1"))

    def test_pending_reply_protects_context_until_send_confirmation(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(
            context,
            active_config(bot_aliases=["机器人"], group_ttl_seconds=60.0),
        )
        event = FakeGroupEvent("机器人，帮我看看？")

        asyncio.run(_collect_with_request_hook(runtime, event))
        state = runtime.context_store.peek_group("group-1")
        self.assertIsNotNone(state)
        state.updated_at -= 120.0
        runtime.record_llm_response(event, FakeLLMResponse("我看看。"))
        event.result = FakeResult([Plain("我看看。")])
        event._has_send_oper = True
        runtime.record_after_message_sent(event)

        restored = runtime.context_store.peek_group("group-1")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.last_user_message().text, "机器人，帮我看看？")
        self.assertEqual(restored.last_bot_message().text, "我看看。")

    def test_pending_protection_respects_max_tracked_groups_hard_limit(self) -> None:
        async def scenario() -> QianjiLingqueRuntime:
            context = FakeContext([])
            runtime = QianjiLingqueRuntime(
                context,
                active_config(
                    bot_aliases=["机器人"],
                    max_tracked_groups=1,
                    score_threshold_reply=0.1,
                    score_threshold_ignore=0.0,
                ),
            )
            first = FakeGroupEvent("这个怎么办？帮忙看看", group_id="group-1")
            await _collect_with_request_hook(runtime, first)
            second = FakeGroupEvent("这个怎么办？帮忙看看", group_id="group-2")
            await _collect_with_request_hook(runtime, second)
            return runtime

        runtime = asyncio.run(scenario())

        self.assertLessEqual(len(runtime.context_store._groups), 1)
        self.assertIsNone(runtime.context_store.peek_group("group-1"))
        self.assertIsNotNone(runtime.context_store.peek_group("group-2"))

    def test_command_message_is_not_added_to_context(self) -> None:
        context = FakeContext([])
        runtime = QianjiLingqueRuntime(context, active_config())
        event = FakeGroupEvent("/天气 北京")

        asyncio.run(_collect(runtime.handle_group_message(event)))
        state = runtime.context_store.peek_group("group-1")

        self.assertIsNone(state)


async def _collect(generator) -> list[object]:
    return [item async for item in generator]


async def _collect_with_request_hook(runtime: QianjiLingqueRuntime, event: FakeGroupEvent) -> list[object]:
    generator = runtime.handle_group_message(event)
    results: list[object] = []
    try:
        first = await generator.__anext__()
    except StopAsyncIteration:
        return results
    results.append(first)
    if event.llm_requests:
        _record_request(runtime, event)
    try:
        async for item in generator:
            results.append(item)
    finally:
        await generator.aclose()
    return results


def _record_request(runtime: QianjiLingqueRuntime, event: FakeGroupEvent) -> None:
    request = event.llm_requests[-1]
    event.set_extra("provider_request", request)
    runtime.record_llm_request(event, request)


if __name__ == "__main__":
    unittest.main()

