from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class AstrBotEntrypointSmokeTests(unittest.TestCase):
    def test_main_imports_and_registers_chinese_commands_when_sdk_is_available(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        sdk_path = repo_root / ".venv" / "Lib" / "site-packages"
        if not (sdk_path / "astrbot").exists():
            self.skipTest("AstrBot SDK is not available in the local isolated test path")

        script = textwrap.dedent(
            """
            import main
            import asyncio
            import sys
            from astrbot.api.provider import ProviderRequest
            from astrbot.core.star.filter.event_message_type import EventMessageTypeFilter
            from astrbot.core.star.filter.permission import PermissionTypeFilter
            from astrbot.core.star.filter.platform_adapter_type import PlatformAdapterTypeFilter
            from astrbot.core.star.star_handler import star_handlers_registry
            from qianji_lingque.config import PluginConfig
            from qianji_lingque.runtime import QianjiLingqueRuntime

            handlers = [
                item for item in star_handlers_registry._handlers
                if item.handler_module_path == "main"
            ]
            names = {item.handler_name for item in handlers}
            assert "on_group_message" in names
            assert "on_llm_response" in names
            assert "on_llm_request" in names
            assert "on_agent_begin" in names
            assert "after_message_sent" in names
            assert "read_air" in names
            assert {"lingque_enable", "lingque_disable", "lingque_mode"} <= names

            group_handler = next(item for item in handlers if item.handler_name == "read_air")
            group_filter = group_handler.event_filters[0]
            assert group_filter.group_name == "读空气"
            assert "空气" in group_filter.alias
            complete_names = set(group_filter.get_complete_command_names())
            assert "读空气" in complete_names
            assert "空气" in complete_names

            event_handler = next(item for item in handlers if item.handler_name == "on_group_message")
            assert any(isinstance(item, EventMessageTypeFilter) for item in event_handler.event_filters)
            assert any(isinstance(item, PlatformAdapterTypeFilter) for item in event_handler.event_filters)
            assert event_handler.extras_configs["priority"] == -sys.maxsize

            for handler_name in ("lingque_enable", "lingque_disable", "lingque_mode"):
                handler = next(item for item in handlers if item.handler_name == handler_name)
                assert any(isinstance(item, PermissionTypeFilter) for item in handler.event_filters)
                command_filter = next(item for item in handler.event_filters if hasattr(item, "get_complete_command_names"))
                names = set(command_filter.get_complete_command_names())
                assert any(name.startswith("读空气 ") for name in names)
                assert any(name.startswith("空气 ") for name in names)

            class Conv:
                cid = "conv-1"
                persona_id = ""
                history = "[]"

            class ConvManager:
                async def get_curr_conversation_id(self, umo):
                    return "conv-1"

                async def new_conversation(self, umo, platform_id=""):
                    return "conv-1"

                async def get_conversation(self, umo, cid):
                    return Conv()

            class Context:
                conversation_manager = ConvManager()

            class Event:
                unified_msg_origin = "aiocqhttp:GroupMessage:group-1"

                def __init__(self):
                    self.stopped = False
                    self.extra = {}

                def get_group_id(self): return "group-1"
                def get_message_str(self): return "机器人，帮我看看这个怎么弄？"
                def get_sender_id(self): return "user-1"
                def get_sender_name(self): return "群友"
                def get_self_id(self): return "bot-1"
                def get_messages(self):
                    class At:
                        qq = "bot-1"
                    return [At()]
                def get_platform_id(self): return "aiocqhttp"
                def get_platform_name(self): return "aiocqhttp"
                def stop_event(self): self.stopped = True
                def should_call_llm(self, call_llm): self.call_llm = call_llm
                def set_extra(self, key, value): self.extra[key] = value
                def get_extra(self, key): return self.extra.get(key)
                def request_llm(self, prompt, system_prompt="", conversation=None, image_urls=None, audio_urls=None, **kwargs):
                    return ProviderRequest(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        conversation=conversation,
                        image_urls=image_urls or [],
                        audio_urls=audio_urls or [],
                    )

            async def collect(gen):
                return [item async for item in gen]

            event = Event()
            runtime = QianjiLingqueRuntime(Context(), PluginConfig(enabled_groups=["*"], bot_aliases=["机器人"]))
            results = asyncio.run(collect(runtime.handle_group_message(event)))
            assert len(results) == 1
            assert isinstance(results[0], ProviderRequest)
            assert event.stopped

            print("astrbot entrypoint ok")
            """,
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(sdk_path)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        self.assertIn("astrbot entrypoint ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
