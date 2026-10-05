import base64
import httpx
import json
import uuid
import wave
from pathlib import Path
from typing import Any

from utilize import (
    encode_audio_to_base64_utf8,
    encode_audio_to_base64_ascii,
    load_speaker_embedding,
    build_instruction_payload,
    is_error_response,
)
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


def _get(args, key: str, default: Any = None) -> Any:
    """Safe attribute access so handlers never explode on missing args.

    Args:
        args: 参数命名空间（SimpleNamespace 等）。
        key: 要读取的属性名。
        default: 属性不存在时的回退值。

    Returns:
        属性值；属性不存在时返回 default。
    """
    return getattr(args, key, default)


def _ensure_wav_file(file_path: str) -> None:
    """检查本地文件是否为 .wav 格式，否则抛出 ValueError。"""
    if not str(file_path).lower().endswith(".wav"):
        raise ValueError(
            f"参考音频必须为 .wav 格式，当前文件: {file_path}"
        )


def _encode_audio_ref(ref_audio, audio_encode: str = "utf-8", require_wav: bool = True) -> str | None:
    """URL / data-URI 值直接透传；本地文件 base64 编码（默认强制 .wav，可放宽）。"""
    if ref_audio is None:
        return None

    # ★ 防御性处理：如果传入的是列表（AstrBot file 配置），取第一个有效元素
    if isinstance(ref_audio, (list, tuple)):
        for item in ref_audio:
            if item:
                ref_audio = item
                break
        else:
            return None

    # ★ 确保转为字符串，避免 WindowsPath 对象在某些环境下不兼容
    ref_audio = str(ref_audio)

    if ref_audio.startswith(("http://", "https://", "data:")):
        return ref_audio
    if require_wav:
        _ensure_wav_file(ref_audio)
    if audio_encode == "utf-8":
        return encode_audio_to_base64_utf8(ref_audio)
    if audio_encode == "ascii":
        return encode_audio_to_base64_ascii(ref_audio)
    raise ValueError(f"Unknown audio_encode: {audio_encode}")


def _encode_audio_ref_list(ref_audio, audio_encode: str = "utf-8", require_wav: bool = True) -> str | list[str] | None:
    """同 _encode_audio_ref，但支持列表（用于多说话人）"""
    if ref_audio is None:
        return None
    if isinstance(ref_audio, (list, tuple)):
        resolved: list[str] = []
        for item in ref_audio:
            value = _encode_audio_ref(item, audio_encode, require_wav)
            if value is not None:
                resolved.append(value)
        return resolved[0] if len(resolved) == 1 else resolved
    return _encode_audio_ref(ref_audio, audio_encode, require_wav)


# 流式响应是裸 PCM，采样率取自 text_to_speech/README.md 中各模型的官方说明。
# 未收录的模型采样率未知，会回退为非流式，避免输出变速音频；也可用 stream_sample_rate 覆盖。
_STREAM_PCM_RATES = {
    "Qwen3-TTS-CustomVoice": 24000,
    "Qwen3-TTS-VoiceDesign": 24000,
    "Qwen3-TTS-Base": 24000,
    "qwen3_tts": 24000,
    "GLM-TTS": 24000,
    "glm_tts": 24000,
    "Fish Speech S2 Pro": 44100,
    "fish_speech": 44100,
    "MOSS-TTS-Nano": 48000,
    "moss_tts_nano": 48000,
}


def _stream_sample_rate(args) -> int:
    """获取流式 PCM 的采样率。

    优先使用 stream_sample_rate 配置，其次查询内置模型表。

    Args:
        args: 参数命名空间（SimpleNamespace 等）。

    Returns:
        采样率（Hz）；无法确定时返回 0。
    """
    override = int(_get(args, "stream_sample_rate", 0) or 0)
    if override > 0:
        return override
    return _STREAM_PCM_RATES.get(str(_get(args, "use_model", "") or ""), 0)


def _speech_request(args, payload: dict) -> str:
    """共享 POST /v1/audio/speech 请求（支持流式），成功返回输出文件路径，失败抛出异常。"""
    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    timeout = float(_get(args, "timeout") or 300.0)

    # 输出目录由插件层给出（位于 AstrBot 临时目录下，由核心自动清理）
    output_dir = Path(_get(args, "output_dir") or (Path(__file__).parent / "temp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # 仅当 payload 确实要求流式（即该模型支持）时才按流式读取响应
    streaming = bool(payload.get("stream"))
    if streaming:
        output_path = output_dir / f"{uuid.uuid4().hex}.pcm"
    else:
        fmt = str(_get(args, "response_format", "wav") or "wav").lower()
        ext = fmt if fmt in ("wav", "mp3", "flac", "pcm", "aac", "opus") else "wav"
        output_path = output_dir / f"{uuid.uuid4().hex}.{ext}"

    if streaming:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", api_url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    error_text = resp.read().decode()
                    logger.error(f"Error {resp.status_code}: {error_text}")
                    raise RuntimeError(f"TTS 服务返回错误 {resp.status_code}: {error_text}")
                total_bytes = 0
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
                        total_bytes += len(chunk)
                logger.info(f"Streamed {total_bytes} bytes to: {output_path}")
                # 流式响应为裸 PCM，补上 WAV 头后才能被消息平台发送
                sample_rate = _stream_sample_rate(args)
                if sample_rate <= 0:
                    raise RuntimeError(
                        "未知流式 PCM 采样率，无法封装为 WAV；"
                        "请在插件配置中填写 stream_sample_rate"
                    )
                wav_path = output_path.with_suffix(".wav")
                with wave.open(str(wav_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(output_path.read_bytes())
                output_path.unlink(missing_ok=True)
                logger.info(f"Wrapped PCM into WAV: {wav_path} ({sample_rate} Hz)")
                return str(wav_path)

    with httpx.Client(timeout=timeout) as client:
        response = client.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        logger.error(f"Error {response.status_code}: {response.text}")
        raise RuntimeError(f"TTS 服务返回错误 {response.status_code}: {response.text}")

    error = is_error_response(response.content)
    if error:
        logger.error(f"Error response: {error}")
        raise RuntimeError(f"TTS 服务返回错误: {error}")

    with open(output_path, "wb") as f:
        f.write(response.content)
    logger.info(f"Audio saved to: {output_path}")
    return str(output_path)


def _chat_request(args, payload: dict) -> bytes:
    """共享 POST /v1/chat/completions 请求（SoulX-Singer），成功返回音频字节，失败抛出异常。"""
    api_url = f"{args.api_base}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    timeout = float(_get(args, "timeout") or 1800.0)

    logger.info(f"Model: {_get(args, 'model', 'unknown')}")
    logger.info("Generating audio via chat completions...")

    with httpx.Client(timeout=timeout) as client:
        response = client.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        logger.error(f"Error {response.status_code}: {response.text}")
        raise RuntimeError(f"Chat completions 返回错误 {response.status_code}: {response.text}")

    try:
        body = response.json()
    except json.JSONDecodeError:
        raise RuntimeError("响应不是有效的 JSON")

    for choice in body.get("choices", []):
        audio_obj = choice.get("message", {}).get("audio")
        if isinstance(audio_obj, dict) and audio_obj.get("data"):
            try:
                return base64.b64decode(audio_obj["data"])
            except Exception as exc:
                raise RuntimeError(f"解码音频失败: {exc}")
    raise RuntimeError("响应中未包含音频数据")


# ── payload builders (one per model family) ──────────────────────────


def _common_speech_payload(args) -> dict:
    """所有 /v1/audio/speech 模型共享的基础 payload。"""
    payload = {
        "input": args.text,
        "response_format": _get(args, "response_format", "wav"),
    }
    # 流式输出不支持 speed（与官方客户端 build_payload 保持一致）
    if not _get(args, "stream", False):
        speed = _get(args, "speed", 1.0)
        if speed != 1.0:
            payload["speed"] = speed
    initial_chunk = _get(args, "initial_codec_chunk_frames")
    if initial_chunk is not None:
        payload["initial_codec_chunk_frames"] = initial_chunk
    return payload


def _build_voice_clone_common(args, payload: dict, audio_encode: str = "utf-8") -> None:
    """添加 ref_audio / ref_text（ref_audio 强制 .wav）。"""
    ref_audio = _encode_audio_ref(_get(args, "ref_audio"), audio_encode)
    if ref_audio:
        payload["ref_audio"] = ref_audio
    ref_text = _get(args, "ref_text")
    if ref_text:
        payload["ref_text"] = ref_text


# -------- 各模型具体构建函数 --------
def _payload_cosyvoice3(args) -> dict:
    payload = _common_speech_payload(args)
    ref_audio = _encode_audio_ref(_get(args, "ref_audio"), "utf-8")
    if not ref_audio:
        raise ValueError("CosyVoice3 requires ref_audio for voice cloning")
    payload["ref_audio"] = ref_audio
    ref_text = _get(args, "ref_text")
    if not ref_text:
        raise ValueError("CosyVoice3 requires ref_text for voice cloning")
    payload["ref_text"] = ref_text
    if _get(args, "stream", False):
        payload["stream"] = True
        payload["stream_format"] = _get(args, "stream_format", "audio")
        payload["response_format"] = "pcm"
    return payload


def _payload_fish_speech(args) -> dict:
    payload = _common_speech_payload(args)
    _build_voice_clone_common(args, payload)
    if _get(args, "stream", False):
        payload["stream"] = True
        payload["stream_format"] = _get(args, "stream_format", "audio")
        payload["response_format"] = "pcm"
    return payload


def _payload_glm_tts(args) -> dict:
    payload = _common_speech_payload(args)
    payload["voice"] = "default"
    ref_audio = _get(args, "ref_audio")
    if not ref_audio:
        raise ValueError("GLM-TTS requires ref_audio for voice cloning")
    ref_audio_str = str(ref_audio)
    if not (ref_audio_str.startswith(("http://", "https://", "data:"))):
        _ensure_wav_file(ref_audio_str)
    payload["ref_audio"] = ref_audio_str
    ref_text = _get(args, "ref_text")
    if not ref_text:
        raise ValueError("GLM-TTS requires ref_text for voice cloning")
    payload["ref_text"] = ref_text
    if _get(args, "stream", False):
        payload["stream"] = True
        payload["stream_format"] = _get(args, "stream_format", "audio")
        payload["response_format"] = "pcm"
    max_new_tokens = _get(args, "max_new_tokens")
    if max_new_tokens:
        payload["max_new_tokens"] = max_new_tokens
    return payload


def _payload_higgs_v2(args) -> dict:
    payload = _common_speech_payload(args)
    _build_voice_clone_common(args, payload, audio_encode="ascii")
    max_new_tokens = _get(args, "max_new_tokens", 300)
    seed = _get(args, "seed", 42)
    payload["max_new_tokens"] = max_new_tokens
    payload["seed"] = seed
    return payload


def _payload_higgs_v3(args) -> dict:
    payload = _common_speech_payload(args)
    _build_voice_clone_common(args, payload, audio_encode="ascii")
    max_new_tokens = _get(args, "max_new_tokens", 2048)
    seed = _get(args, "seed", 42)
    payload["max_new_tokens"] = max_new_tokens
    payload["seed"] = seed
    return payload


def _payload_indextts2(args) -> dict:
    payload = _common_speech_payload(args)
    voice = _get(args, "voice")
    ref_audio = _encode_audio_ref(_get(args, "ref_audio"), "utf-8")
    # 对照官方 speech_client.py：voice 与 ref_audio 至少提供一个
    if not voice and not ref_audio:
        raise ValueError("IndexTTS-2 requires ref_audio or voice for voice cloning")
    if voice:
        payload["voice"] = voice
    if ref_audio:
        payload["ref_audio"] = ref_audio

    extra_params = {}
    emo_audio = _get(args, "emo_audio")
    if emo_audio:
        emo_audio_str = str(emo_audio)
        if emo_audio_str.startswith(("http://", "https://", "data:")):
            extra_params["emo_audio"] = emo_audio_str
        else:
            extra_params["emo_audio"] = encode_audio_to_base64_utf8(emo_audio_str)
    emo_text = _get(args, "emo_text")
    if emo_text:
        extra_params["emo_text"] = emo_text
    emo_vector = str(_get(args, "emo_vector") or "").strip()
    if emo_vector:
        # 配置中为逗号分隔的 8 维向量，需要转换成数值列表再上传
        values = [v.strip() for v in emo_vector.replace("，", ",").split(",") if v.strip()]
        if len(values) != 8:
            raise ValueError(
                f"IndexTTS-2: emo_vector 需要 8 个逗号分隔的数值，当前 {len(values)} 个"
            )
        try:
            extra_params["emo_vector"] = [float(v) for v in values]
        except ValueError as exc:
            raise ValueError("IndexTTS-2: emo_vector 只能包含数值") from exc
    emo_alpha = _get(args, "emo_alpha")
    if emo_alpha is not None:
        extra_params["emo_alpha"] = emo_alpha
    if _get(args, "use_emo_text", False):
        extra_params["use_emo_text"] = True
    if _get(args, "use_random", False):
        extra_params["use_random"] = True
    if extra_params:
        payload["extra_params"] = extra_params
    return payload


def _validate_ming_args(args) -> None:
    """对照官方 ming_tts/openai_speech_client.py 的 validate_args，尽早拦截非法组合。"""
    ref_audio = _get(args, "ref_audio")
    ref_audio_count = len(ref_audio) if isinstance(ref_audio, (list, tuple)) else (1 if ref_audio else 0)
    if _get(args, "ref_text") and ref_audio_count == 0:
        raise ValueError("Ming-omni-tts: 提供 ref_text 时必须同时提供 ref_audio")
    if _get(args, "speaker_embedding") and ref_audio_count > 1:
        raise ValueError("Ming-omni-tts: speaker_embedding 不能与多个 ref_audio 同时使用")


def _payload_ming_tts(args) -> dict:
    _validate_ming_args(args)
    payload = _common_speech_payload(args)
    voice = _get(args, "voice")
    if voice:
        payload["voice"] = voice
    task_type = _get(args, "task_type")
    if task_type:
        payload["task_type"] = task_type
    dialect = _get(args, "dialect")
    if dialect:
        payload["language"] = dialect

    instructions = build_instruction_payload(args)
    if instructions:
        payload["instructions"] = instructions

    ref_audio = _encode_audio_ref_list(_get(args, "ref_audio"), "utf-8", require_wav=False)
    if ref_audio:
        payload["ref_audio"] = ref_audio
    ref_text = _get(args, "ref_text")
    if ref_text:
        payload["ref_text"] = ref_text
    speaker_embedding = _get(args, "speaker_embedding")
    if speaker_embedding:
        # ★ 防御性处理：确保是字符串路径
        se_path = speaker_embedding
        if isinstance(se_path, (list, tuple)):
            se_path = next((s for s in se_path if s), None)
        if se_path:
            payload["speaker_embedding"] = load_speaker_embedding(str(se_path))
    max_new_tokens = _get(args, "max_new_tokens")
    if max_new_tokens:
        payload["max_new_tokens"] = max_new_tokens
    if _get(args, "stream", False):
        payload["stream"] = True
        payload["stream_format"] = _get(args, "stream_format", "audio")
        payload["response_format"] = "pcm"
    return payload


def _payload_ming_flash(args) -> dict:
    payload = _common_speech_payload(args)
    instructions = build_instruction_payload(args)
    if instructions:
        payload["instructions"] = instructions
    return payload


def _payload_moss_tts_nano(args) -> dict:
    payload = _common_speech_payload(args)
    ref_audio = _encode_audio_ref(_get(args, "ref_audio"), "utf-8")
    if not ref_audio:
        raise ValueError("MOSS-TTS-Nano requires ref_audio for voice cloning")
    payload["ref_audio"] = ref_audio
    if _get(args, "stream", False):
        payload["stream"] = True
        payload["stream_format"] = _get(args, "stream_format", "audio")
        payload["response_format"] = "pcm"
    max_new_tokens = _get(args, "max_new_tokens")
    if max_new_tokens:
        payload["max_new_tokens"] = max_new_tokens
    return payload


def _payload_omnivoice(args) -> dict:
    payload = _common_speech_payload(args)
    voice = _get(args, "voice")
    if voice:
        payload["voice"] = voice
    language = _get(args, "language")
    if language:
        payload["language"] = language
    ref_audio = _encode_audio_ref(_get(args, "ref_audio"), "utf-8")
    if ref_audio:
        payload["ref_audio"] = ref_audio
    ref_text = _get(args, "ref_text")
    if ref_text:
        payload["ref_text"] = ref_text
    instructions = _get(args, "instructions")
    if instructions:
        payload["instructions"] = instructions
    seed = _get(args, "seed")
    if seed is not None:
        payload["extra_params"] = {"seed": seed}
    if _get(args, "stream", False):
        payload["stream"] = True
        payload["stream_format"] = _get(args, "stream_format", "audio")
        payload["response_format"] = "pcm"
    return payload


# Qwen3-TTS 各模型对应的默认 task_type（与官方客户端语义一致）
_QWEN3_TASK_TYPES = {
    "Qwen3-TTS-CustomVoice": "CustomVoice",
    "Qwen3-TTS-VoiceDesign": "VoiceDesign",
    "Qwen3-TTS-Base": "Base",
}


def _payload_qwen3_tts(args) -> dict:
    payload = _common_speech_payload(args)

    # 未显式配置时按所选模型补全，避免用错任务类型
    task_type = _get(args, "task_type") or _QWEN3_TASK_TYPES.get(
        _get(args, "use_model"), "CustomVoice"
    )
    payload["task_type"] = task_type

    # ★ 关键：必须调用 _encode_audio_ref 将文件路径转为 Data URL
    ref_audio = _encode_audio_ref(_get(args, "ref_audio"), "utf-8")
    logger.info(f"Encoded ref_audio type: {type(ref_audio)}, length: {len(ref_audio) if ref_audio else 0}")
    if ref_audio:
        payload["ref_audio"] = ref_audio

    ref_text = _get(args, "ref_text")
    if ref_text:
        payload["ref_text"] = ref_text

    if _get(args, "language"):
        payload["language"] = _get(args, "language")
    if _get(args, "instructions"):
        payload["instructions"] = _get(args, "instructions")
    if _get(args, "max_new_tokens"):
        payload["max_new_tokens"] = _get(args, "max_new_tokens")

    speaker_embedding = _get(args, "speaker_embedding")
    if speaker_embedding:
        # ★ 防御性处理：确保是字符串路径再 open
        se_path = speaker_embedding
        if isinstance(se_path, (list, tuple)):
            se_path = next((s for s in se_path if s), None)
        if se_path:
            with open(str(se_path), encoding="utf-8") as f:
                payload["speaker_embedding"] = json.load(f)

    # 对照官方客户端：开启后才发送 x_vector_only_mode
    if _get(args, "x_vector_only", False):
        payload["x_vector_only_mode"] = True

    non_streaming = _get(args, "non_streaming_mode")
    if non_streaming is not None:
        payload["non_streaming_mode"] = non_streaming
    else:
        payload["non_streaming_mode"] = True

    if _get(args, "stream", False):
        payload["stream"] = True
        payload["stream_format"] = _get(args, "stream_format", "audio")
        payload["response_format"] = "pcm"

    # ★ 预设音色：服务端规范字段为 voice，历史配置中的 speaker 作为兼容别名
    voice = _get(args, "voice") or _get(args, "speaker")
    if voice:
        payload["voice"] = voice

    # 对照官方客户端 tts_common.build_payload：各任务类型的必填项校验
    if task_type == "VoiceDesign" and not payload.get("instructions"):
        raise ValueError("Qwen3-TTS VoiceDesign requires instructions to design a voice")
    if task_type == "Base" and not (
        ref_audio or payload.get("speaker_embedding") or payload.get("voice")
    ):
        raise ValueError(
            "Qwen3-TTS Base requires ref_audio, speaker_embedding or a precomputed voice"
        )

    return payload


def _payload_voxcpm2(args) -> dict:
    payload = _common_speech_payload(args)
    payload["voice"] = "default"
    # 官方客户端使用 "(描述)文本" 前缀实现音色设计 / 可控克隆
    control_instruction = str(_get(args, "control_instruction") or "").strip()
    if control_instruction:
        payload["input"] = f"({control_instruction}){args.text}"
    ref_audio = _encode_audio_ref(_get(args, "ref_audio"), "utf-8")
    if ref_audio:
        payload["ref_audio"] = ref_audio
    return payload


def _payload_voxtral_tts(args) -> dict:
    payload = _common_speech_payload(args)
    voice = _get(args, "voice")
    if voice:
        payload["voice"] = voice
    ref_audio = _encode_audio_ref(_get(args, "ref_audio"), "utf-8")
    if ref_audio:
        payload["ref_audio"] = ref_audio
    ref_text = _get(args, "ref_text")
    if ref_text:
        payload["ref_text"] = ref_text
    # 对照 voxtral_tts/gradio_demo.py：CFG 强度通过 extra_params.cfg_alpha 上传
    cfg_alpha = _get(args, "cfg_alpha")
    if cfg_alpha is not None:
        payload["extra_params"] = {"cfg_alpha": cfg_alpha}
    if _get(args, "stream", False):
        payload["stream"] = True
        payload["stream_format"] = _get(args, "stream_format", "audio")
        payload["response_format"] = "pcm"
    return payload


def _payload_soulxsinger(args) -> dict:
    extra_args = dict(_get(args, "extra_args", {}) or {})
    extra_args.setdefault("vocal_sep", _get(args, "vocal_sep", False))
    extra_args.setdefault("auto_shift", _get(args, "auto_shift", True))
    extra_args.setdefault("pitch_shift", 0)

    prompt_audio = _get(args, "prompt_audio")
    target_audio = _get(args, "target_audio")
    if prompt_audio:
        prompt_audio_str = str(prompt_audio)
        if not (prompt_audio_str.startswith(("http://", "https://", "data:"))):
            _ensure_wav_file(prompt_audio_str)
        extra_args["prompt_audio"] = prompt_audio_str
    if target_audio:
        target_audio_str = str(target_audio)
        if not (target_audio_str.startswith(("http://", "https://", "data:"))):
            _ensure_wav_file(target_audio_str)
        extra_args["target_audio"] = target_audio_str
    preprocess = _get(args, "preprocess_weights_dir")
    if preprocess:
        extra_args["preprocess_weights_dir"] = preprocess

    language = _get(args, "language", "Mandarin")
    if not _get(args, "svc", False):
        extra_args["language"] = language
        extra_args["control"] = _get(args, "control", "melody")

    content: list[dict] = [{"type": "text", "text": "soulx-singer"}]
    prompt_data = _get(args, "prompt_audio_data_url")
    if prompt_data:
        prompt_data = str(prompt_data)
        if prompt_data.startswith(("http://", "https://")):
            # input_audio.data 只接受内联音频（参见官方客户端 openai_chat_client.py），远程 URL 先下载再编码
            response = httpx.get(prompt_data, timeout=float(_get(args, "timeout") or 300.0))
            response.raise_for_status()
            prompt_data = (
                "data:audio/wav;base64,"
                + base64.b64encode(response.content).decode("ascii")
            )
        elif not prompt_data.startswith("data:"):
            _ensure_wav_file(prompt_data)
            prompt_data = encode_audio_to_base64_ascii(prompt_data)
        content.append({"type": "input_audio", "input_audio": {"data": prompt_data, "format": "mp3"}})

    payload = {
        "modalities": ["audio"],
        "messages": [{"role": "user", "content": content}],
        "num_inference_steps": _get(args, "num_inference_steps", 32),
        "guidance_scale": _get(args, "guidance_scale", 3.0),
        "extra_args": extra_args,
    }
    seed = _get(args, "seed")
    if seed is not None:
        payload["seed"] = seed
    return payload


# ── dispatch ─────────────────────────────────────────────────────────

TTS_HANDLERS = {
    "CosyVoice3": _payload_cosyvoice3,
    "cosyvoice3": _payload_cosyvoice3,
    "Fish Speech S2 Pro": _payload_fish_speech,
    "fish_speech": _payload_fish_speech,
    "GLM-TTS": _payload_glm_tts,
    "glm_tts": _payload_glm_tts,
    "higgs_audio_v2": _payload_higgs_v2,
    "Higgs Audio v2": _payload_higgs_v2,
    "higgs_audio_v3": _payload_higgs_v3,
    "Higgs Audio v3": _payload_higgs_v3,
    "IndexTTS-2": _payload_indextts2,
    "indextts2": _payload_indextts2,
    "Ming-omni-tts": _payload_ming_tts,
    "ming_tts": _payload_ming_tts,
    "Ming-flash-omni-TTS": _payload_ming_flash,
    "ming_flash_omni_tts": _payload_ming_flash,
    "MOSS-TTS-Nano": _payload_moss_tts_nano,
    "moss_tts_nano": _payload_moss_tts_nano,
    "OmniVoice": _payload_omnivoice,
    "omnivoice": _payload_omnivoice,
    "Qwen3-TTS-CustomVoice": _payload_qwen3_tts,
    "Qwen3-TTS-VoiceDesign": _payload_qwen3_tts,
    "Qwen3-TTS-Base": _payload_qwen3_tts,
    "qwen3_tts": _payload_qwen3_tts,
    "VoxCPM2": _payload_voxcpm2,
    "voxcpm2": _payload_voxcpm2,
    "Voxtral TTS": _payload_voxtral_tts,
    "voxtral_tts": _payload_voxtral_tts,
    "SoulX-Singer": _payload_soulxsinger,
    "soulxsinger": _payload_soulxsinger,
}


def run_tts(args) -> str:
    """统一入口：分发到对应模型处理器并发送请求，成功返回音频文件路径，失败抛出异常。"""
    use_model = _get(args, "use_model")
    if not use_model:
        raise ValueError("args.use_model is required to select the TTS model")

    handler = TTS_HANDLERS.get(use_model)
    if handler is None:
        raise ValueError(
            f"Unknown use_model: {use_model!r}. "
            f"Supported models: {sorted(set(TTS_HANDLERS))}"
        )

    payload = handler(args)

    # 流式 PCM 采样率未知时回退非流式：裸 PCM 猜错采样率会变速，宁可不用流式
    if payload.get("stream") and _stream_sample_rate(args) <= 0:
        logger.warning(
            "%s 的流式 PCM 采样率未知，本次请求回退为非流式；"
            "如确认采样率，可在插件配置中填写 stream_sample_rate。",
            use_model,
        )
        args.stream = False
        payload = handler(args)

    # 该模型不支持流式时 payload 中不会出现 stream 字段，此处回退为非流式并提示
    if (
        _get(args, "stream", False)
        and not payload.get("stream")
        and use_model not in ("SoulX-Singer", "soulxsinger")
    ):
        logger.warning("%s 不支持流式输出，本次请求已回退为非流式。", use_model)

    if use_model in ("SoulX-Singer", "soulxsinger"):
        audio = _chat_request(args, payload)
        # SoulX-Singer 走 chat completions，返回的是完整 WAV 字节
        output_dir = Path(_get(args, "output_dir") or (Path(__file__).parent / "temp"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{uuid.uuid4().hex}.wav"
        with open(output_path, "wb") as f:
            f.write(audio)
        logger.info(f"Audio saved to: {output_path}")
        return str(output_path)

    try:
        return _speech_request(args, payload)
    except Exception as exc:
        if not payload.get("stream"):
            raise
        logger.warning("流式请求失败（%s），自动回退为非流式重试。", exc)
        args.stream = False
        return _speech_request(args, handler(args))
