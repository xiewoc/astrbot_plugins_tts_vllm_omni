import base64
import httpx
import json
import uuid
from pathlib import Path

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


def _get(args, key, default=None):
    """Safe attribute access so handlers never explode on missing args."""
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


def _speech_request(args, payload: dict) -> str:
    """共享 POST /v1/audio/speech 请求（支持流式），成功返回输出文件路径，失败抛出异常。"""
    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    timeout = float(_get(args, "timeout") or 300.0)

    plugin_dir = Path(__file__).parent / "temp"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    if _get(args, "stream", False):
        output_path = plugin_dir / f"{uuid.uuid4().hex}.pcm"
    else:
        fmt = str(_get(args, "response_format", "wav") or "wav").lower()
        ext = fmt if fmt in ("wav", "mp3", "flac", "pcm", "aac", "opus") else "wav"
        output_path = plugin_dir / f"{uuid.uuid4().hex}.{ext}"

    if _get(args, "stream", False):
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
                return str(output_path)

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
    if voice:
        payload["voice"] = voice
    ref_audio = _encode_audio_ref(_get(args, "ref_audio"), "utf-8")
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
    emo_vector = _get(args, "emo_vector")
    if emo_vector is not None:
        extra_params["emo_vector"] = emo_vector
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


def _payload_qwen3_tts(args) -> dict:
    payload = _common_speech_payload(args)

    task_type = _get(args, "task_type", "Base")
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

    non_streaming = _get(args, "non_streaming_mode")
    if non_streaming is not None:
        payload["non_streaming_mode"] = non_streaming
    else:
        payload["non_streaming_mode"] = True

    if _get(args, "stream", False):
        payload["stream"] = True
        payload["stream_format"] = _get(args, "stream_format", "audio")
        payload["response_format"] = "pcm"

    # ★ 只添加用户明确指定的 voice/speaker（不为空时）
    voice = _get(args, "voice")
    if voice:
        payload["voice"] = voice
    speaker = _get(args, "speaker")
    if speaker:
        payload["speaker"] = speaker

    return payload


def _payload_voxcpm2(args) -> dict:
    payload = _common_speech_payload(args)
    payload["voice"] = "default"
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
        if not prompt_data.startswith("data:"):
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

    if use_model in ("SoulX-Singer", "soulxsinger"):
        audio = _chat_request(args, payload)
        output_path = Path(__file__).parent / f"{uuid.uuid4().hex}.wav"
        with open(output_path, "wb") as f:
            f.write(audio)
        logger.info(f"Audio saved to: {output_path}")
        return str(output_path)

    return _speech_request(args, payload)