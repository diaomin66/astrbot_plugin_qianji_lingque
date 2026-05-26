from __future__ import annotations

import unittest

from qianji_lingque.config import PluginConfig, normalize_mode, parse_mode
from qianji_lingque.event_utils import snapshot_from_event
from qianji_lingque.runtime import QianjiLingqueRuntime


class FakeConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved = False

    def save_config(self):
        self.saved = True


class FakeEvent:
    def __init__(
        self,
        group_id: str = "group-1",
        text: str = "你好",
        *,
        platform_id: str = "",
        platform_name: str = "",
    ) -> None:
        self.group_id = group_id
        self.message_str = text
        self.platform_id = platform_id
        self.platform_name = platform_name
        self.unified_msg_origin = f"{platform_id}:GroupMessage:{group_id}" if platform_id else ""

    def get_group_id(self) -> str:
        return self.group_id

    def get_message_str(self) -> str:
        return self.message_str

    def get_sender_id(self) -> str:
        return "user-1"

    def get_sender_name(self) -> str:
        return "用户"

    def get_self_id(self) -> str:
        return "bot-1"

    def get_messages(self) -> list[object]:
        return []

    def get_platform_id(self) -> str:
        return self.platform_id

    def get_platform_name(self) -> str:
        return self.platform_name


class ConfigRuntimeTests(unittest.TestCase):
    def test_parse_mode_rejects_unknown_input(self) -> None:
        self.assertIsNone(parse_mode("胡来"))
        self.assertEqual(normalize_mode("胡来"), "normal")
        self.assertEqual(parse_mode("积极"), "active")

    def test_disable_group_uses_blacklist_when_all_groups_enabled(self) -> None:
        raw = FakeConfig({"enabled_groups": ["*"], "disabled_groups": []})
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)

        runtime.disable_group(FakeEvent("group-1"))

        self.assertEqual(config.enabled_groups, ["*"])
        self.assertEqual(config.disabled_groups, ["group-1"])
        self.assertFalse(config.is_group_enabled("group-1"))
        self.assertTrue(config.is_group_enabled("group-2"))
        self.assertEqual(raw["disabled_groups"], ["group-1"])
        self.assertTrue(raw.saved)

    def test_empty_enabled_groups_means_no_passive_listening_by_default(self) -> None:
        config = PluginConfig.from_astrbot_config({"enabled_groups": [], "disabled_groups": []})

        self.assertFalse(config.is_group_enabled("group-1"))

    def test_disable_group_does_not_empty_whitelist_into_global_enable(self) -> None:
        raw = FakeConfig({"enabled_groups": ["group-1"], "disabled_groups": []})
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)

        runtime.disable_group(FakeEvent("group-1"))

        self.assertEqual(config.enabled_groups, ["group-1"])
        self.assertEqual(config.disabled_groups, ["group-1"])
        self.assertFalse(config.is_group_enabled("group-1"))
        self.assertFalse(config.is_group_enabled("group-2"))

    def test_enable_group_removes_blacklist_entry(self) -> None:
        raw = FakeConfig({"enabled_groups": [], "disabled_groups": ["group-1"]})
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)

        runtime.enable_group(FakeEvent("group-1"))

        self.assertEqual(config.enabled_groups, ["group-1"])
        self.assertEqual(config.disabled_groups, [])
        self.assertTrue(config.is_group_enabled("group-1"))

    def test_enable_group_removes_legacy_platform_blacklist_entry(self) -> None:
        raw = FakeConfig(
            {
                "enabled_groups": ["*"],
                "disabled_groups": ["onebot-a:group-1"],
            },
        )
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)

        runtime.enable_group(
            FakeEvent(
                "group-1",
                platform_id="onebot-a",
                platform_name="aiocqhttp",
            ),
        )

        self.assertEqual(config.disabled_groups, [])

    def test_enable_scoped_group_keeps_bare_disabled_safety_entry(self) -> None:
        raw = FakeConfig(
            {
                "enabled_groups": ["*"],
                "disabled_groups": ["group-1", "onebot-a:group-1"],
            },
        )
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)

        runtime.enable_group(
            FakeEvent(
                "group-1",
                platform_id="onebot-a",
                platform_name="aiocqhttp",
            ),
        )

        self.assertEqual(config.disabled_groups, ["group-1"])

    def test_enable_scoped_group_appends_scoped_key_when_bare_enabled_exists(self) -> None:
        raw = FakeConfig({"enabled_groups": ["group-1"], "disabled_groups": []})
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)
        event = FakeEvent(
            "group-1",
            platform_id="onebot-a",
            platform_name="aiocqhttp",
        )

        message = runtime.enable_group(event)

        self.assertIn("已开启当前群", message)
        self.assertEqual(config.enabled_groups, ["group-1", "onebot-a:GroupMessage:group-1"])
        self.assertIn("群开关：开启", runtime.render_status(event))

    def test_enable_scoped_group_warns_when_bare_disabled_still_blocks(self) -> None:
        raw = FakeConfig({"enabled_groups": ["*"], "disabled_groups": ["group-1"]})
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)
        event = FakeEvent(
            "group-1",
            platform_id="onebot-a",
            platform_name="aiocqhttp",
        )

        message = runtime.enable_group(event)

        self.assertIn("安全阀仍生效", message)
        self.assertEqual(config.disabled_groups, ["group-1"])
        self.assertIn("群开关：关闭", runtime.render_status(event))

    def test_unsupported_platform_command_does_not_mutate_config(self) -> None:
        raw = FakeConfig({"enabled_groups": [], "disabled_groups": []})
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)

        message = runtime.enable_group(
            FakeEvent(
                "group-1",
                platform_id="telegram-a",
                platform_name="telegram",
            ),
        )

        self.assertIn("只支持 aiocqhttp", message)
        self.assertEqual(config.enabled_groups, [])
        self.assertFalse(raw.saved)

    def test_enable_group_reports_global_disabled_state(self) -> None:
        raw = FakeConfig({"enabled": False, "enabled_groups": [], "disabled_groups": ["group-1"]})
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)

        message = runtime.enable_group(FakeEvent("group-1"))

        self.assertIn("总开关已关闭", message)
        self.assertEqual(config.disabled_groups, ["group-1"])

    def test_mode_command_does_not_silently_default_unknown_input(self) -> None:
        raw = FakeConfig({"mode": "normal"})
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)

        message = runtime.set_mode(FakeEvent("group-1"), "不存在")

        self.assertIn("可切换为", message)
        self.assertEqual(config.mode, "normal")

    def test_mode_command_requires_group_context(self) -> None:
        raw = FakeConfig({"mode": "normal"})
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)

        message = runtime.set_mode(FakeEvent(""), "积极")

        self.assertIn("请在群聊中使用", message)
        self.assertEqual(config.group_modes, {})

    def test_mode_command_sets_group_mode(self) -> None:
        raw = FakeConfig({"mode": "普通", "group_modes": {}})
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)

        message = runtime.set_mode(FakeEvent("group-1"), "积极")

        self.assertIn("积极", message)
        self.assertEqual(config.effective_mode("group-1"), "active")
        self.assertEqual(raw["group_modes"], ["group-1=active"])

    def test_group_modes_accept_legacy_dict_and_hidden_list_storage(self) -> None:
        legacy = PluginConfig.from_astrbot_config({"group_modes": {" group-1 ": "积极"}})
        current = PluginConfig.from_astrbot_config({"group_modes": [" group-2 =quiet"]})

        self.assertEqual(legacy.effective_mode("group-1"), "active")
        self.assertEqual(current.effective_mode("group-2"), "quiet")

    def test_save_writes_full_normalized_config(self) -> None:
        raw = FakeConfig({"mode": "普通", "bot_aliases": ["机器人"]})
        config = PluginConfig.from_astrbot_config(raw)

        config.save()

        self.assertEqual(raw["max_context_messages"], 24)
        self.assertEqual(raw["max_tracked_groups"], 200)
        self.assertEqual(raw["group_ttl_seconds"], 86400.0)
        self.assertEqual(raw["score_threshold_reply"], 0.78)
        self.assertEqual(raw["bot_aliases"], ["机器人"])
        self.assertTrue(raw["llm_gate_enabled"])
        self.assertEqual(raw["llm_gate_timeout_seconds"], 1.5)
        self.assertEqual(raw["llm_gate_fallback_score"], 0.4)
        self.assertTrue(raw["log_decisions_enabled"])
        self.assertFalse(raw["log_message_excerpt_enabled"])
        self.assertTrue(raw.saved)

    def test_log_settings_default_on_and_message_excerpt_opt_in(self) -> None:
        default_config = PluginConfig.from_astrbot_config({})
        enabled_config = PluginConfig.from_astrbot_config(
            {
                "log_decisions_enabled": False,
                "log_message_excerpt_enabled": True,
                "llm_gate_enabled": False,
                "llm_gate_timeout_seconds": 0.1,
                "llm_gate_fallback_score": 1.5,
            },
        )

        self.assertTrue(default_config.llm_gate_enabled)
        self.assertEqual(default_config.llm_gate_timeout_seconds, 1.5)
        self.assertTrue(default_config.log_decisions_enabled)
        self.assertFalse(default_config.log_message_excerpt_enabled)
        self.assertFalse(enabled_config.llm_gate_enabled)
        self.assertEqual(enabled_config.llm_gate_timeout_seconds, 0.2)
        self.assertEqual(enabled_config.llm_gate_fallback_score, 1.0)
        self.assertFalse(enabled_config.log_decisions_enabled)
        self.assertTrue(enabled_config.log_message_excerpt_enabled)

    def test_save_trims_group_mode_keys(self) -> None:
        raw = FakeConfig({"group_modes": {}})
        config = PluginConfig.from_astrbot_config(raw)
        config.group_modes[" group-1 "] = "active"

        config.save()

        self.assertEqual(raw["group_modes"], ["group-1=active"])

    def test_group_command_reports_private_context(self) -> None:
        raw = FakeConfig({"enabled_groups": [], "disabled_groups": []})
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)

        message = runtime.disable_group(FakeEvent(""))

        self.assertIn("请在群聊中使用", message)

    def test_last_reason_reads_group_state(self) -> None:
        raw = FakeConfig({"enabled_groups": [], "disabled_groups": []})
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)
        state = runtime.context_store.get_group("group-1")
        state.last_decision = "ignore (0.10)：测试原因"

        message = runtime.render_last_reason(FakeEvent("group-1"))

        self.assertIn("测试原因", message)

    def test_last_reason_does_not_create_group_state(self) -> None:
        raw = FakeConfig({"enabled_groups": [], "disabled_groups": []})
        config = PluginConfig.from_astrbot_config(raw)
        runtime = QianjiLingqueRuntime(context=None, config=config)

        message = runtime.render_last_reason(FakeEvent("group-1"))

        self.assertIn("还没有判定记录", message)
        self.assertEqual(runtime.context_store._groups, {})

    def test_last_reason_respects_debug_switch(self) -> None:
        config = PluginConfig.from_astrbot_config({"debug_explain_enabled": False})
        runtime = QianjiLingqueRuntime(context=None, config=config)

        message = runtime.render_last_reason(FakeEvent("group-1"))

        self.assertIn("原因查看已关闭", message)

    def test_last_reason_requires_group_context(self) -> None:
        config = PluginConfig.from_astrbot_config({})
        runtime = QianjiLingqueRuntime(context=None, config=config)

        message = runtime.render_last_reason(FakeEvent(""))

        self.assertIn("请在群聊中使用", message)

    def test_disabled_group_message_does_not_create_context(self) -> None:
        config = PluginConfig.from_astrbot_config({"disabled_groups": ["group-1"]})
        runtime = QianjiLingqueRuntime(context=None, config=config)

        async def collect() -> list[object]:
            return [item async for item in runtime.handle_group_message(FakeEvent("group-1"))]

        self.assertEqual(asyncio_run(collect()), [])
        self.assertEqual(runtime.context_store._groups, {})

    def test_bad_list_and_threshold_config_is_normalized(self) -> None:
        config = PluginConfig.from_astrbot_config(
            {
                "enabled_groups": [" group-2 ", ""],
                "disabled_groups": "group-1",
                "score_threshold_ignore": 0.9,
                "score_threshold_reply": 0.2,
                "merge_window_seconds": -10,
                "reply_timeout_seconds": float("nan"),
                "takeover_explicit_mentions": "off",
                "enabled": "maybe",
                "bot_aliases": [" 机器人 ", ""],
            },
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.enabled_groups, ["group-2"])
        self.assertEqual(config.disabled_groups, ["group-1"])
        self.assertEqual(config.score_threshold_ignore, 0.2)
        self.assertEqual(config.score_threshold_reply, 0.9)
        self.assertEqual(config.merge_window_seconds, 0.0)
        self.assertEqual(config.reply_timeout_seconds, 12.0)
        self.assertFalse(config.takeover_explicit_mentions)
        self.assertEqual(config.bot_aliases, ["机器人"])

    def test_status_uses_distinct_group_switch_label(self) -> None:
        config = PluginConfig.from_astrbot_config({"mode": "normal", "enabled_groups": ["group-1"]})
        runtime = QianjiLingqueRuntime(context=None, config=config)

        message = runtime.render_status(FakeEvent("group-1"))

        self.assertIn("当前群：group-1", message)
        self.assertIn("群开关：开启", message)

    def test_status_requires_group_context(self) -> None:
        config = PluginConfig.from_astrbot_config({"enabled_groups": ["*"]})
        runtime = QianjiLingqueRuntime(context=None, config=config)

        message = runtime.render_status(FakeEvent(""))

        self.assertIn("请在群聊中使用", message)
        self.assertNotIn("群开关：开启", message)

    def test_status_reports_unsupported_platform_without_enabled_state(self) -> None:
        config = PluginConfig.from_astrbot_config({"enabled_groups": ["*"]})
        runtime = QianjiLingqueRuntime(context=None, config=config)

        message = runtime.render_status(
            FakeEvent(
                "group-1",
                platform_id="telegram-a",
                platform_name="telegram",
            ),
        )

        self.assertIn("只支持 aiocqhttp", message)
        self.assertNotIn("群开关：开启", message)

    def test_ascii_alias_requires_word_boundary(self) -> None:
        event = FakeEvent("group-1", text="robotics is fun")

        snap = snapshot_from_event(event, ["bot"])

        self.assertFalse(snap.mentions_bot)

    def test_chinese_alias_requires_wake_boundary(self) -> None:
        generic = snapshot_from_event(FakeEvent("group-1", text="机器人学习挺难"), ["机器人"])
        common_wake = snapshot_from_event(FakeEvent("group-1", text="机器人可以做到吗"), ["机器人"])
        compact_wake = snapshot_from_event(FakeEvent("group-1", text="机器人你看下"), ["机器人"])
        direct = snapshot_from_event(FakeEvent("group-1", text="机器人，帮我看看"), ["机器人"])

        self.assertFalse(generic.mentions_bot)
        self.assertFalse(common_wake.mentions_bot)
        self.assertTrue(compact_wake.mentions_bot)
        self.assertTrue(direct.mentions_bot)


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
