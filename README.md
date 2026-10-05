# astrbot_plugins_tts_vllm_omni

一个基于 [vLLM-Omni](https://github.com/vllm-project/vllm-omni) 的 AstrBot 文字转语音（TTS）插件。

通过 OpenAI 兼容的 `/v1/audio/speech` 接口与 vLLM-Omni 在线服务通信，将 Qwen3-TTS、Ming-omni-tts、Higgs Audio、IndexTTS-2、SoulX-Singer 等主流 TTS 模型无缝接入 AstrBot，支持音色克隆、情感控制、流式输出与随机语音回复。

## 特性

- 🎙️ **多模型支持**：内置 15+ 种主流 TTS 模型的分发器，切换模型只需改一个配置项
- 🔊 **音色克隆**：通过 `ref_audio` + `ref_text`（或 `speaker_embedding`）克隆任意音色
- 🤖 **LLM 工具调用**：注册 `tts_speech` / `send_vocal_msg_no_return` 工具，用户要求"发语音/朗读"时由 Agent 自动调用
- 🎲 **随机 TTS**：按概率将纯文本回复替换为语音，让机器人"开口说话"
- 🌊 **流式输出**：支持 PCM 流式输出（依模型而定），返回前自动补 WAV 头，保证平台可播放
- 🗂️ **临时文件治理**：音频统一写入 AstrBot 临时目录，由核心按 `temp_dir_max_size` 自动清理
- 🎭 **情感控制**：IndexTTS-2 情感参考音频 / 情感文本 / 强度调节
- 🌏 **多语言与方言**：支持 Chinese / English / French 等语言及粤语等方言控制（Ming 系列）
- 🎵 **歌声合成**：SoulX-Singer 人声 / 旋律 / 伴奏条件合成

## 支持的模型

| 模型（use_model） | HuggingFace 仓库 | 音色克隆 | 流式 | 关键参数 |
|---|---|---|---|---|
| `Qwen3-TTS-CustomVoice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | — | ✓ | `voice` 预设音色 |
| `Qwen3-TTS-VoiceDesign` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | — | ✓ | `instructions` 描述生成音色 |
| `Qwen3-TTS-Base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | ✓ | ✓ | `ref_audio` + `ref_text` / `speaker_embedding` |
| `Ming-omni-tts` | `inclusionAI/Ming-omni-tts-0.5B` | ✓ | ✓ | 多参考音频、`dialect` 方言、192 维 `speaker_embedding` |
| `Ming-flash-omni-TTS` | `Jonathan1909/Ming-flash-omni-2.0` | — | — | `instructions` 字幕/风格控制 |
| `Higgs Audio v2` | `higgsaudio/higgs-audio-v2` | ✓ | — | `ascii` 编码、`seed` 确定性输出 |
| `Higgs Audio v3` | `higgsaudio/higgs-audio-v3` | ✓ | — | `ascii` 编码、`seed` 确定性输出 |
| `IndexTTS-2` | `IndexTeam/IndexTTS-2` | ✓ | — | `emo_*` 情感控制、`voice` 上传音色 |
| `Fish Speech S2 Pro` | `fishaudio/s2-pro` | ✓ | ✓ | `ref_audio` + `ref_text` |
| `GLM-TTS` | `zai-org/GLM-TTS` | ✓ | ✓ | 必须 `ref_audio` + `ref_text`（需在配置中手动输入模型名） |
| `Voxtral TTS` | `mistralai/Voxtral-4B-TTS-2603` | ✓ | ✓ | `voice` 预设、`ref_audio` + `ref_text` |
| `CosyVoice3` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | ✓ | ✓ | 必须 `ref_audio` + `ref_text` |
| `OmniVoice` | `k2-fsa/OmniVoice` | ✓ | ✓ | `voice` / `language` / `instructions` |
| `VoxCPM2` | `openbmb/VoxCPM2` | ✓ | — | `voice`（默认 `default`） |
| `MOSS-TTS-Nano` | `OpenMOSS-Team/MOSS-TTS-Nano` | ✓ | ✓ | 必须 `ref_audio`（忽略 `ref_text`） |
| `SoulX-Singer` | `Soul-AILab/SoulX-Singer` | ✓ | — | `prompt_audio` / `target_audio`（走 `/v1/chat/completions`） |

> [!NOTE]
> `use_model` 只决定插件本地如何构造请求参数；实际加载的模型以 vLLM-Omni 服务启动参数为准。请保证两者一致。

## 环境要求

- 已部署并运行 [vLLM-Omni](https://github.com/vllm-project/vllm-omni) TTS 服务（需 GPU）
- AstrBot v3.4.x+（插件使用 `Star`、`FunctionTool`、`filter.on_decorating_result` 等 API）
- 平台支持：`aiocqhttp`（如 NapCat、Lagrange）、`qq_official`

## 快速开始

### 1. 部署 vLLM-Omni TTS 服务

以 Qwen3-TTS 为例（端口默认 `8091`，可用 `--port` 修改）：

```bash
vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --omni \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --trust-remote-code \
    --port 8091
```

其他模型的启动示例见插件内 `text_to_speech/README.md` 及各模型子目录（内含启动脚本、Gradio Demo、客户端示例）：

- Ming：`text_to_speech/ming_tts/README.md`
- Higgs Audio v3：`text_to_speech/higgs_audio_v3/README.md`

### 2. 安装插件

在 AstrBot WebUI「插件管理」中搜索安装，或将本仓库克隆到 `data/plugins/` 目录后重启 AstrBot。

### 3. 配置插件

在「插件管理」→「插件配置」中填写：

**基础设置（basic_config）**

| 配置项 | 说明 | 默认值 |
|---|---|---|
| `base_url` | vLLM 服务器地址（不含端口） | `http://localhost` |
| `port` | vLLM 服务端口 | `8091` |
| `api_key` | API 密钥，vLLM-Omni 默认无需鉴权 | `EMPTY` |
| `timeout` | 请求超时时间（秒），长文本建议调大 | `300.0` |
| `if_random_tts` | 是否开启随机语音回复 | `false` |
| `random_tts_factor` | 随机因子，越大越易触发（范围建议 0 ~ 1） | `0.4` |
| `max_concurrency` | 同时进行的 TTS 请求上限，单卡建议 1；超出时随机语音自动跳过 | `1` |
| `retry_times` | 请求失败重试次数（0.5s、1s… 退避；配置类错误不重试） | `1` |

**模型与音色设置（model_config + 各模型配置分节）**

`model_config` 只保留全局项，其余参数按模型拆分到独立分节（如 `voxtral_tts`、`qwen3_base`、`soulxsinger`）。WebUI 会根据 `use_model` 自动显示当前模型的分节，分节中的同名键优先级高于 `model_config`。

| 配置项 | 说明 | 默认值 |
|---|---|---|
| `use_model` | 模型分发器选择（同时决定显示哪个模型分节） | `Qwen3-TTS-CustomVoice` |
| `model` | 模型名称（仅本地日志记录） | 空 |
| `response_format` | 输出格式：`wav` / `mp3` / `flac` / `pcm` / `aac` / `opus` | `wav` |
| `stream` | 启用流式输出（启用后 `response_format` 强制为 `pcm`，返回前自动补 WAV 头） | `false` |
| `stream_format` | 流式分片格式，官方示例固定为 `audio` | `audio` |
| `stream_sample_rate` | 流式 PCM 采样率（Hz），`0` = 使用内置模型表；未知采样率时自动回退非流式 | `0` |

各模型分节内的字段与「支持的模型」表格中的关键参数一一对应，例如：

| 分节 | 主要字段 |
|---|---|
| `qwen3_custom_voice` | `voice`（预设音色，如 `vivian` 等）、`language`、`instructions`、`max_new_tokens`、`non_streaming_mode` |
| `qwen3_voice_design` | `instructions`（必填）、`language`、`max_new_tokens`、`non_streaming_mode` |
| `qwen3_base` | `ref_audio`、`ref_text`、`speaker_embedding`、`x_vector_only`、`voice`、`language`、`max_new_tokens`、`non_streaming_mode` |
| `ming_tts` | `ref_audio`（可多文件）、`ref_text`、`voice`、`dialect`、`instructions` / `instruction_json`、`speaker_embedding` |
| `indextts2` | `voice`、`emo_audio`、`emo_text`、`emo_alpha`、`use_emo_text`、`use_random`、`emo_vector` |
| `soulxsinger` | `prompt_audio` / `prompt_audio_data_url`、`target_audio`、`language`、`control`、`svc`、`vocal_sep`、`auto_shift` |

> [!TIP]
> `ref_audio`、`speaker_embedding`、`emo_audio`、`prompt_audio`、`target_audio`、`prompt_audio_data_url` 等文件类字段，可填写相对于 `data/plugin_data/astrbot_plugins_tts_vllm_omni/` 的相对路径，也可以直接填写 `http(s)` / `data:` URL。插件启动时会自动校验参考音频是否存在。

## 使用方式

插件提供三种触发路径，可同时启用：

### 1. Agent 自动调用（推荐）

插件向 LLM Agent 注册了 `tts_speech` 工具。当用户说出"用语音回复""读给我听""发一段语音"等指令时，模型会自动调用该工具合成语音并发送。

### 2. 语音消息（不返回文本）

注册了 `send_vocal_msg_no_return` 工具，用于仅发送语音、不附带文本的场景。

### 3. 随机 TTS

开启 `if_random_tts` 后，机器人每条纯文本回复会以 `random_tts_factor` 的概率被替换为语音。因子越大越易触发，设置为 `0` 可关闭。合成失败时自动回退为文本。

## 模型使用指南

### Qwen3-TTS（CustomVoice / VoiceDesign / Base）

- **CustomVoice**：`use_model=Qwen3-TTS-CustomVoice`，在 `qwen3_custom_voice` 分节填写 `voice`（如 `vivian`、`ryan`、`aiden`，可用音色可通过 `<服务器>/v1/audio/voices` 查询）使用预设音色
- **VoiceDesign**：`use_model=Qwen3-TTS-VoiceDesign`，在 `instructions` 中描述期望的音色特征
- **Base（音色克隆）**：`use_model=Qwen3-TTS-Base`，提供 `ref_audio` + `ref_text`，或将预计算好的说话人嵌入 JSON 填入 `speaker_embedding`；需要关闭 ICL 模式时开启 `x_vector_only`

### Ming-omni-tts

支持**多参考音频**（多说话人播客风格）：`ref_audio` 可配置多个文件。可通过 `dialect` 控制方言（如 `广粤话`）、通过 `voice` 指定 IP 音色标签，或通过 `speaker_embedding` 使用 192 维嵌入。注意 `speaker_embedding` 与多个 `ref_audio` 不能同时使用。

### Higgs Audio v2 / v3

参考音频由插件以 `ascii` base64 编码后上传（否则服务端会解析失败），使用 `seed` 可复现确定性输出。

### IndexTTS-2

情感控制三件套：`emo_audio`（情感参考音频）、`emo_text`（情感描述）、`emo_alpha`（强度 0~1）。开启 `use_emo_text` 可从文本自动推断情感，开启 `use_random` 使用随机情感原型。

### SoulX-Singer（歌声合成）

走 `/v1/chat/completions` 接口。通过 `prompt_audio` 提供人声/旋律条件音频，通过 `target_audio` 提供目标伴奏，实现指定伴奏的歌声合成。`prompt_audio_data_url` 用于内联提示音频：本地 `.wav` 与 `http(s)` URL 都会由插件编码成 data URI 后随请求发送（适合插件与服务器不在同一台机器）。

## 常见问题

**Q：请求超时 / 报错 `Error 401/404`？**</br>
A：确认 `base_url` 与 `port` 指向的地址可访问（`curl http://<host>:<port>/v1/audio/speech`），并检查防火墙。API 密钥一般填 `EMPTY` 即可。

**Q：提示参考音频格式错误？**</br>
A：多数模型要求 `ref_audio` 为 `.wav` 格式，且大小不超过 10MB、时长不超过 30s。Higgs 系列所需的 `ascii` 编码差异由插件自动处理。

**Q：配置了 `use_model` 但请求失败，报 "Unknown use_model"？**</br>
A：请使用插件内置模型列表中的精确名称（见上文"支持的模型"表格第一列），注意大小写。

**Q：开启 `stream` 后为什么提示回退为非流式？**</br>
A：流式响应是裸 PCM，需要采样率才能补上 WAV 头。插件内置表只收录了已确认的模型（Qwen3 / GLM 为 24000、Fish Speech 为 44100、MOSS-TTS-Nano 为 48000），其他模型请在 `stream_sample_rate` 中填写实际采样率后再启用流式。若服务端不支持流式，插件会自动退回非流式重试。

**Q：随机 TTS 没有生效？**</br>
A：确认已开启 `if_random_tts`，且 `random_tts_factor` 大于 0。该功能只对纯文本回复生效，且要求 TTS 服务可达。

**Q：远程主机拒绝连接？**</br>
A：确认地址可以ping通且防火墙设置了放行规则

## 性能

vllm的性能无需多言

实测在Nvidia Tesla A2下，RTF在0.6~0.7之间，使用参数：

```bash
vllm serve /home/xie/vllm/Qwen3-TTS-12Hz-1.7B-Base --port 8091 --gpu-memory-utilization 0.3 --enforce-eager --max-model-len 2048 --omni
```

## 目录结构

```
astrbot_plugin_tts_vllm_omni/
├── main.py              # 插件主入口：LLM 工具、随机 TTS、事件钩子
├── tts.py               # TTS 请求构建与分发（各模型 handler）
├── utilize.py           # 工具函数（base64 编码、说话人嵌入校验等）
├── _conf_schema.json    # WebUI 配置 Schema
├── metadata.yaml        # 插件元数据
├── scripts/
│   ├── test_tts.py      # Qwen3-TTS 直连测试脚本（计算 RTF）
│   └── qwen3_tts_test_voice_design_single.wav  # 参考音频示例
└── text_to_speech/      # vLLM-Omni 各模型在线服务文档 / 客户端 / 脚本
```

## 致谢

- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)：底层多模态服务框架
- [AstrBot](https://github.com/Soulter/AstrBot)：插件运行框架

当然还有

- DeepSeek: 帮我写了这个README和大部分调试
- xiewoc(俺): 帮助上面那位对抗幻觉

事件原因，有些模型并未测试

如有问题或建议，欢迎在本插件仓库提交 Issue。

~~*DS现在好贵的，都怪小难梁*~~