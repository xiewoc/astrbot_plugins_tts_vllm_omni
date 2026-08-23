"""AstrBot TTS Plugin - vLLM Omni Backend"""

import asyncio
import os
import sys
import types
from pathlib import Path
from random import random

from pydantic import Field
from pydantic.dataclasses import dataclass

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
import astrbot.api.message_components as Comp
from astrbot.api.message_components import Plain, Record

# 确保本地 tts 模块可被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts import run_tts

# Ming 系列模型支持多参考音频
_MULTI_REF_MODELS = {"Ming-omni-tts", "ming_tts"}

# 需要从 model_config 中单独处理的 file 类型字段
_FILE_FIELDS = frozenset({
    "ref_audio", "speaker_embedding", "emo_audio", "prompt_audio", "target_audio",
})


# ---------------------------------------------------------------------------
# TTS Tool
# ---------------------------------------------------------------------------

@dataclass
class TTSTool(FunctionTool[AstrAgentContext]):
    """TTS 语音合成工具，供 LLM Agent 调用。"""

    name: str = "tts_speech"
    description: str = "TTS 语音合成。当用户要求朗读/发语音时调用，返回音频文件路径。"
    parameters: dict = Field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "需要合成语音的文本内容。"},
        },
        "required": ["text"],
    })
    config: dict = Field(default_factory=dict)
    base_data_path: Path = Field(default_factory=Path)

    # ---- 安全路径解析 ----

    def _safe_resolve_path(self, item) -> str | None:
        """安全地解析文件路径，防止路径遍历攻击。

        确保返回的路径严格位于 self.base_data_path 内部。
        如果传入值为空或非法，返回 None。
        """
        if item is None:
            return None
        if isinstance(item, (list, tuple)):
            for sub in item:
                result = self._safe_resolve_path(sub)
                if result is not None:
                    return result
            return None
        if not isinstance(item, (str, Path)):
            return None

        item_str = str(item).strip()
        if not item_str:
            return None

        try:
            candidate = (self.base_data_path / item_str).resolve()
            base_resolved = self.base_data_path.resolve()
            # 关键安全检查：解析后的路径必须在 base_data_path 之下
            candidate.relative_to(base_resolved)
        except (OSError, ValueError):
            logger.warning(
                "安全拦截: 路径 '%s' 试图访问数据目录 '%s' 之外的文件，已拒绝。",
                item_str, self.base_data_path,
            )
            return None

        return str(candidate)

    # ---- 内部工具方法 ----

    def _resolve_file_field(self, value) -> str | None:
        """将 AstrBot file 类型配置（始终为 list）解包为单个字符串路径或 None。"""
        return self._safe_resolve_path(value)

    def _resolve_ref_audio(self, raw_ref_audio, use_model: str) -> str | list[str] | None:
        """处理 ref_audio：Ming 系列保留多文件列表，其余取首个。"""
        if isinstance(raw_ref_audio, (list, tuple)):
            paths = []
            for item in raw_ref_audio:
                resolved = self._safe_resolve_path(item)
                if resolved is not None:
                    paths.append(resolved)
            logger.debug("ref_audio resolved paths: %s", paths)
            if use_model in _MULTI_REF_MODELS and len(paths) > 1:
                return paths
            return paths[0] if paths else None

        resolved = self._safe_resolve_path(raw_ref_audio)
        if resolved is not None:
            logger.debug("ref_audio resolved path: %s", resolved)
        return resolved

    def _build_args(self, text: str) -> types.SimpleNamespace:
        """根据插件配置构建 TTS 请求参数命名空间。"""
        basic = dict(self.config.get("basic_config", {}))
        model = dict(self.config.get("model_config", {}))
        args = types.SimpleNamespace()

        # -- 基础连接参数 --
        base_url = str(basic.get("base_url", "http://localhost")).rstrip("/")
        args.api_base = f"{base_url}:{basic.get('port', '8091')}"
        args.api_key = basic.get("api_key", "EMPTY")
        args.timeout = float(basic.get("timeout", 300.0) or 300.0)
        args.text = text
        args.model = model.get("model") or model.get("use_model", "unknown")

        # -- File 类型字段（全部通过安全路径解析）--
        args.ref_audio = self._resolve_ref_audio(model.get("ref_audio"), args.model)
        args.speaker_embedding = self._resolve_file_field(model.get("speaker_embedding"))
        args.emo_audio = self._resolve_file_field(model.get("emo_audio"))
        args.prompt_audio = self._resolve_file_field(model.get("prompt_audio"))
        args.target_audio = self._resolve_file_field(model.get("target_audio"))

        # -- 模型参数（批量赋值，保持与 schema 顺序一致）--
        _model_params = {
            "task_type": "CustomVoice",
            "speaker": "",
            "voice": "",
            "language": "Chinese",
            "response_format": "wav",
            "stream": False,
            "max_new_tokens": 300,
            "seed": 42,
            "instructions": "",
            "ref_text": "",
            "dialect": "",
            "instruction_json": "",
            "x_vector_only": False,
            "emo_text": "",
            "emo_alpha": 0.0,
            "use_emo_text": False,
            "use_random": False,
            "non_streaming_mode": True,
        }
        for key, default in _model_params.items():
            setattr(args, key, model.get(key, default))

        # -- 透传其余未知键（跳过已处理的 file 字段，防止列表覆盖）--
        for key, value in model.items():
            if key not in _FILE_FIELDS and not hasattr(args, key):
                setattr(args, key, value)

        return args

    # ---- 对外调用接口 ----

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        """LLM Agent 工具调用入口。"""
        text = (kwargs.get("text") or "").strip()
        if not text:
            return "Missing 'text' in input"
        try:
            args = self._build_args(text)
            opt_path = await asyncio.to_thread(run_tts, args)
            if not opt_path:
                raise RuntimeError("TTS 服务未返回音频路径")
            logger.info("TTS 合成成功: %s", opt_path)
            return f"语音合成完成，保存为 {Path(opt_path).name}"
        except Exception as exc:
            logger.error("TTS 合成失败: %s", exc, exc_info=True)
            return f"语音合成失败：{exc}"

    async def etr_call(self, text: str) -> Path:
        """事件触发调用入口，返回音频文件 Path 对象。"""
        args = self._build_args(text)
        opt_path = await asyncio.to_thread(run_tts, args)
        if not opt_path:
            raise RuntimeError("TTS 服务未返回音频路径")
        logger.info("TTS 合成成功: %s", opt_path)
        return Path(opt_path)


# ---------------------------------------------------------------------------
# Random TTS
# ---------------------------------------------------------------------------

class RandomTTS:
    """按概率将纯文本消息链替换为语音消息。"""

    def __init__(self, tts_tool: TTSTool):
        self._tts_tool = tts_tool

    async def maybe_replace(self, chain: list, factor: float) -> list:
        """若随机命中且消息链为纯文本，则尝试转为语音；失败时回退为原文本。"""
        if random() > factor:
            return chain

        # 仅对纯文本消息链生效
        if not all(isinstance(e, Plain) for e in chain):
            return chain

        text = "".join(msg.text for msg in chain if isinstance(msg, Plain))
        if not text:
            return chain

        logger.info("Random TTS triggered (factor=%.3f, threshold=%.3f)", random(), factor)
        try:
            wav_path = await self._tts_tool.etr_call(text)
            return [Record(file=str(wav_path))]
        except Exception as exc:
            logger.error("Random TTS failed, falling back to text: %s", exc)
            return [Plain(text=text)]


# ---------------------------------------------------------------------------
# Plugin Entry
# ---------------------------------------------------------------------------

@register("astrbot_plugin_tts_vllm_omni", "xiewoc", "https://github.com/xiewoc", "1.0.1")
class AstrBot_Plugin_tts_vllm_omni(Star):
    """vLLM-Omni TTS 插件主入口。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        base_data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name
        self.tts_tool = TTSTool(config=config, base_data_path=base_data_path)
        self.context.add_llm_tools(self.tts_tool)

        # 随机 TTS 相关配置延迟到 initialize 中读取
        self._random_tts: RandomTTS | None = None
        self._if_random_tts: bool = False
        self._random_tts_factor: float = 0.4

    async def initialize(self):
        data_dir = self.tts_tool.base_data_path
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("TTS 插件初始化完成，数据目录: %s", data_dir)

        # 读取随机 TTS 配置
        basic_cfg = self.config.get("basic_config", {})
        self._if_random_tts = basic_cfg.get("if_random_tts", False)
        self._random_tts_factor = float(basic_cfg.get("random_tts_factor", 0.4))
        self._random_tts = RandomTTS(self.tts_tool)

        # 检查参考音频是否存在（使用安全路径解析）
        model_cfg = self.config.get("model_config", {})
        ref_audio = model_cfg.get("ref_audio")
        if ref_audio:
            items = ref_audio if isinstance(ref_audio, (list, tuple)) else [ref_audio]
            for idx, item in enumerate(items):
                resolved = self.tts_tool._safe_resolve_path(item)
                if resolved is None:
                    label = f"#{idx}" if isinstance(ref_audio, (list, tuple)) else ""
                    logger.warning("参考音频%s路径无效或被安全策略拦截: %s", label, item)
                elif not Path(resolved).exists():
                    label = f"#{idx}" if isinstance(ref_audio, (list, tuple)) else ""
                    logger.warning("参考音频%s不存在: %s", label, resolved)

        # 安全打印配置（脱敏）
        safe_config = {
            "basic": {k: v for k, v in basic_cfg.items() if k != "api_key"},
            "model": {k: v for k, v in model_cfg.items() if k != "ref_audio"},
        }
        logger.debug("TTS 插件配置: %s", safe_config)

    async def terminate(self):
        logger.info("TTS 插件已终止。")

    # ---- 事件处理器 ----

    @filter.llm_tool(name="send_vocal_msg_no_return")
    async def send_vocal_msg_no_return(self, event: AstrMessageEvent, text: str) -> MessageEventResult:
        """发送语音消息（不返回文本）。仅在用户明确要求发语音时使用。

        Args:
            text(string): 需要合成为语音的文本内容
        """
        path = await self.tts_tool.etr_call(text)
        yield event.chain_result([Comp.Record(file=str(path))])

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        """结果装饰钩子：按概率将纯文本回复替换为语音。"""
        if not self._if_random_tts or self._random_tts is None:
            return
        chain = event.get_result().chain
        event.get_result().chain = await self._random_tts.maybe_replace(
            chain, self._random_tts_factor,
        )
