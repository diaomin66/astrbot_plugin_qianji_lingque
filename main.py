from __future__ import annotations

import sys
from typing import Any, AsyncIterator

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

if __package__:
    from .qianji_lingque.config import PluginConfig
    from .qianji_lingque.runtime import QianjiLingqueRuntime
else:
    from qianji_lingque.config import PluginConfig
    from qianji_lingque.runtime import QianjiLingqueRuntime


class QianjiLingquePlugin(Star):
    """千机聆阙插件入口。"""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = PluginConfig.from_astrbot_config(config)
        self.runtime = QianjiLingqueRuntime(context, self.config)

    async def terminate(self) -> None:
        self.runtime.terminate()

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=-sys.maxsize)
    async def on_group_message(self, event: AstrMessageEvent) -> AsyncIterator[Any]:
        async for result in self.runtime.handle_group_message(event):
            yield result

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse) -> None:
        self.runtime.record_llm_response(event, response)

    @filter.on_agent_begin()
    async def on_agent_begin(self, event: AstrMessageEvent, run_context: Any) -> None:
        self.runtime.prepare_agent_run(event, run_context)

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent) -> None:
        self.runtime.record_after_message_sent(event)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        self.runtime.record_llm_request(event, req)

    @filter.command_group("读空气", alias={"空气"})
    async def read_air(self) -> None:
        pass

    @read_air.command("状态", alias={"查"})
    async def lingque_status(self, event: AstrMessageEvent) -> AsyncIterator[Any]:
        yield event.plain_result(self.runtime.render_status(event))
        event.stop_event()

    @filter.permission_type(filter.PermissionType.ADMIN)
    @read_air.command("开启", alias={"开"})
    async def lingque_enable(self, event: AstrMessageEvent) -> AsyncIterator[Any]:
        yield event.plain_result(self.runtime.enable_group(event))
        event.stop_event()

    @filter.permission_type(filter.PermissionType.ADMIN)
    @read_air.command("关闭", alias={"关"})
    async def lingque_disable(self, event: AstrMessageEvent) -> AsyncIterator[Any]:
        yield event.plain_result(self.runtime.disable_group(event))
        event.stop_event()

    @filter.permission_type(filter.PermissionType.ADMIN)
    @read_air.command("模式", alias={"模"})
    async def lingque_mode(self, event: AstrMessageEvent, mode: str = "") -> AsyncIterator[Any]:
        message = self.runtime.set_mode(event, mode)
        yield event.plain_result(message)
        event.stop_event()

    @read_air.command("原因", alias={"因"})
    async def lingque_reason(self, event: AstrMessageEvent) -> AsyncIterator[Any]:
        yield event.plain_result(self.runtime.render_last_reason(event))
        event.stop_event()
