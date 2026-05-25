from __future__ import annotations

import unittest

from qianji_lingque.config import PluginConfig
from qianji_lingque.context import ContextStore
from qianji_lingque.decision import DecisionEngine
from qianji_lingque.event_utils import MessageSnapshot


def snapshot(
    text: str,
    *,
    group_id: str = "group-1",
    sender_id: str = "user-1",
    timestamp: float = 100.0,
    mentions_bot: bool = False,
    replies_to_bot: bool = False,
) -> MessageSnapshot:
    return MessageSnapshot(
        group_id=group_id,
        sender_id=sender_id,
        sender_name=sender_id,
        message_id=f"msg-{timestamp}",
        self_id="bot-1",
        text=text,
        outline=text,
        timestamp=timestamp,
        mentions_bot=mentions_bot,
        replies_to_bot=replies_to_bot,
    )


class DecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PluginConfig(enabled_groups=["group-1"], bot_aliases=["千机"])
        self.store = ContextStore(max_messages=24)
        self.engine = DecisionEngine()

    def test_direct_mention_replies(self) -> None:
        state = self.store.get_group("group-1")

        decision = self.engine.decide(snapshot("千机这个怎么弄？", mentions_bot=True), state, self.config)

        self.assertEqual(decision.action, "reply")

    def test_command_message_is_ignored(self) -> None:
        state = self.store.get_group("group-1")

        decision = self.engine.decide(snapshot("/天气 北京"), state, self.config)

        self.assertEqual(decision.action, "ignore")
        self.assertIn("指令", decision.reason)

    def test_consecutive_user_messages_wait(self) -> None:
        state = self.store.get_group("group-1")
        state.append_user_message(snapshot("我先说一句", timestamp=100.0))

        decision = self.engine.decide(snapshot("还有一句", timestamp=100.5), state, self.config)

        self.assertEqual(decision.action, "wait")
        self.assertIn("连续发言", decision.reason)

    def test_group_chatter_without_direct_signal_ignores(self) -> None:
        state = self.store.get_group("group-1")
        state.append_user_message(snapshot("今天吃啥", sender_id="user-1", timestamp=90.0))
        state.append_user_message(snapshot("火锅", sender_id="user-2", timestamp=91.0))

        decision = self.engine.decide(snapshot("可以", sender_id="user-3", timestamp=100.0), state, self.config)

        self.assertEqual(decision.action, "ignore")

    def test_gray_area_is_marked(self) -> None:
        state = self.store.get_group("group-1")

        decision = self.engine.decide(snapshot("这个有点怪呢，看看", timestamp=100.0), state, self.config)

        self.assertTrue(decision.is_gray_area)

    def test_recent_bot_message_increases_followup_score(self) -> None:
        state = self.store.get_group("group-1")
        state.append_bot_reply("上一句回复", timestamp=95.0)

        decision = self.engine.decide(snapshot("那这个呢？", timestamp=100.0), state, self.config)

        self.assertIn("接续 bot 发言", decision.reason)

    def test_min_reply_interval_penalizes_non_direct_message(self) -> None:
        state = self.store.get_group("group-1")
        state.append_bot_reply("刚刚回过", timestamp=99.0)

        decision = self.engine.decide(snapshot("随便聊聊", timestamp=100.0), state, self.config)

        self.assertIn("未过最短回复间隔", decision.reason)

    def test_disabled_group_ignores(self) -> None:
        config = PluginConfig(enabled_groups=["group-1"], disabled_groups=["group-1"])
        state = self.store.get_group("group-1")

        decision = self.engine.decide(snapshot("千机？", mentions_bot=True), state, config)

        self.assertEqual(decision.action, "ignore")


if __name__ == "__main__":
    unittest.main()
