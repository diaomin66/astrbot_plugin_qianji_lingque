# 千机聆阙项目计划

## Summary

千机聆阙是一个 AstrBot 群聊读空气插件：旁路监听群消息，但不抢占普通指令和其他插件；先用本地快速规则判断“大概率不该回”的消息，只在明确需要回复时沿用 AstrBot 当前会话模型和人格生成自然回复。

## Architecture

- 插件包名：`astrbot_plugin_qianji_lingque`
- 展示名：`千机聆阙`
- 作者：雪碧bir
- 支持平台：当前版本只声明并处理 `aiocqhttp`
- 监听入口：`@filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)`
- 管理指令：`/读空气`，别名 `/空气`
- 默认安全策略：新安装不自动监听任何群；用 `/读空气 开启` 当前群，或在 `enabled_groups` 显式填 `*` 才全群启用。

## Runtime Flow

1. `FastGate`：本地忽略命令、空消息、bot 自己、连续发言等。
2. `ScoreGate`：本地评分，综合 @/引用、问题意图、求助意图、近期 bot 发言、群友互聊、冷却和模式。
3. `TimingGate`：强求助、常见疑问或接近回复阈值的灰区消息调用一次短超时 LLM 节奏判断，只输出 `reply | wait | ignore`；低置信 `wait/ignore`、失败或超时后，如果本地信号明显像强求助，则兜底接话；弱疑问只进入 TimingGate，不在节奏判断关闭时直接兜底；高置信 `wait/ignore` 会被尊重。
4. `ReplyComposer`：优先通过 AstrBot `request_llm` 生成自然回复，复用当前会话、人设和基础媒体能力；不适合主动 ProviderRequest 的第三方 runner 只在灰区求助时尝试短文本直发兜底。

## Commands

- `/读空气 状态`：查看当前群启用状态和模式。
- `/读空气 开启`：启用当前群。
- `/读空气 关闭`：关闭当前群。
- `/读空气 模式 安静|普通|积极`：调整当前群回复积极度。
- `/读空气 原因`：查看最近一次判定原因。

## Config

- `enabled`: 总开关。
- `enabled_groups`: 显式启用群列表；空列表表示不自动监听，`*` 表示所有群；多实例优先使用 `unified_msg_origin` 或旧的 `平台实例ID:群号`。
- `disabled_groups`: 显式禁用群列表，优先级高于启用列表；裸群号禁用会作用于同群号，作为更保守的安全阀。
- `bot_aliases`: bot 昵称；默认空，避免泛昵称误触发。
- `max_context_messages`: 每群短期上下文条数。
- `max_tracked_groups`: 最多保留群状态数。
- `group_ttl_seconds`: 群状态无活动后的清理时间。
- `takeover_explicit_mentions`: 是否接管 @/引用/wake 场景，避免默认 LLM 重复回答。
- `llm_gate_enabled`: 是否启用灰区轻量 LLM 节奏判断，默认开启；强求助、常见疑问或接近回复阈值的灰区会调用，普通闲聊灰区不会调用。
- `llm_gate_timeout_seconds`: 灰区节奏判断和第三方 runner 直发兜底超时，默认 1.5 秒，超时安全降级或本地兜底。
- `llm_gate_fallback_score`: 灰区本地兜底分数，默认 0.4，用于覆盖约 0.42 的强求助灰区；弱疑问只进入 TimingGate；高置信 TimingGate `wait/ignore` 不会被本地兜底强行覆盖。
- `log_decisions_enabled`: 是否在 AstrBot 日志中输出读空气判定、计划正式回复 LLM、TimingGate 实际调用 LLM、正式回复实际调用 LLM 和发送收尾信息，默认开启但隐藏原文。
- `log_message_excerpt_enabled`: 是否在日志中输出最多 80 字消息摘要，默认关闭，仅建议临时排障使用。

## Performance Strategy

- 只对强求助、常见疑问或接近回复阈值的灰区消息调用轻量 TimingGate；低分消息和普通闲聊灰区不调用模型，高分消息直接进入正式回复。
- TimingGate 默认 1.5 秒超时；低置信 `wait/ignore`、超时、异常或非法 JSON 时，强求助型灰区消息会本地兜底为 `reply`，弱疑问和普通灰区仍 `wait`。
- 每群独立冷却，bot 刚回复后更克制。
- 每群只保留短期环形上下文，且限制总群数和 TTL。
- 复杂媒体、远程媒体场景放行 AstrBot 默认链路；第三方 runner 的灰区求助会用灰区短超时尝试直发兜底，失败后放行默认链路，但默认链路未必会回复。
- 判定日志默认开启但隐藏群聊原文；所有读空气判定和关键 LLM 生命周期使用 INFO，便于用户排障。

## Test Plan

- 单元测试：配置解析、上下文淘汰、门控评分、prompt 安全、媒体白名单。
- 集成烟测：AstrBot SDK 可用时验证 handler 注册、中文命令、ProviderRequest 路径。
- 回归重点：默认不全群监听、旧配置 key 兼容、@/引用/wake 接管、昵称不截断其他插件、pending 保护上下文直到发送确认。

验证命令：

```powershell
python -m compileall -q main.py qianji_lingque tests
python -m json.tool _conf_schema.json > $null
python -m unittest discover -s tests
```
