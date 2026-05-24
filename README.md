# 千机聆阙

作者：雪碧bir

千机聆阙是一个 AstrBot 群聊读空气插件。它旁路监听群消息，先用本地规则和轻量打分判断是否该回复，灰区消息快速降级或交给明确点名场景，最终沿用 AstrBot 当前会话模型和人格生成回复。

## 功能

- 群聊上下文短期记忆
- 本地快速门控，减少无意义模型调用
- 灰区本地跳过，避免在群聊主链路上等待模型
- 使用 AstrBot 标准 `request_llm` 回复管线，保留当前人格与会话
- 已发送/已产出回复后的冷却和连续发言等待
- 中文管理指令

## 指令

- `/读空气 状态`：查看当前群状态
- `/读空气 开启`：启用当前群
- `/读空气 关闭`：关闭当前群
- `/读空气 模式 安静`：更克制
- `/读空气 模式 普通`：平衡模式
- `/读空气 模式 积极`：更积极
- `/读空气 原因`：查看最近一次判定原因

也可以使用别名 `/空气`。

## 配置

插件通过 AstrBot WebUI 读取 `_conf_schema.json`。默认只在群聊生效，`enabled_groups` 留空表示所有群启用。

`wait` 表示本轮先不回复：连续发言、灰区消息、同群已有回复生成中都会走这个降级。消息链 @ 或引用 bot 时，默认会强接管并抑制 AstrBot 默认 LLM 链路；AstrBot wake 前缀触发且插件主动请求回复时，只抑制默认 LLM，不截断其他插件；纯昵称命中不会截断其他插件。可在 WebUI 关闭“接管明确点名”。

当前版本使用 AstrBot 标准 `ProviderRequest` 会话管线，并透传当前消息链里的基础图片和语音。媒体只接管 AstrBot `Plain`、`At`、`Image`、`Record` 和只读 `Reply` 组件；图片/语音引用仅接受 `base64://`、AstrBot 临时目录内已存在的本地路径，或指向临时目录内已存在文件的 `file:///`。为避免 AstrBot ProviderRequest 组装时产生 MIME 错配，本插件主动接管的图片限制为 JPEG，语音限制为 WAV；`base64://` 编码长度限制为 700000 字符，本地媒体大小限制为 1.5MB。无法识别或格式不匹配的媒体会放行默认链路。复杂引用附件、远程媒体 URL、视频/文件解析、未知组件以及 Dify/Coze/DashScope/DeerFlow 等第三方 runner 场景会放行 AstrBot 默认链路；放行场景本身不会追加用户消息到本插件上下文，但默认链路或其他插件随后真实发出的回复会作为 bot 节奏信号进入短期上下文和冷却，避免刚回复完又插话。

当前发布元数据仅声明已验证的 `aiocqhttp`；其他平台可能可用，但不在当前版本承诺范围内。

## 安装

```powershell
cd <AstrBot目录>\data\plugins
git clone https://github.com/diaomin66/astrbot_plugin_qianji_lingque astrbot_plugin_qianji_lingque
```

重启 AstrBot 或在 WebUI 重载插件后，在目标群发送 `/读空气 状态` 做烟测，再按需使用 `/读空气 开启`、`/读空气 模式 普通` 调整。

## 开发验证

```powershell
python -m compileall -q main.py qianji_lingque tests
python -m unittest discover -s tests
```
