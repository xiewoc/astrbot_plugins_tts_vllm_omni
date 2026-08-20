import base64
import os
import json

EXPECTED_SPEAKER_EMBEDDING_DIM = 192  # 默认 Ming 系列维度，Qwen3 可能兼容，若不匹配可传入维度参数


def encode_audio_to_base64_utf8(audio_path: str) -> str:
    """Encode a local audio file to a base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    ext = audio_path.lower().rsplit(".", 1)[-1]
    mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg", "aac": "audio/aac"}
    mime_type = mime_map.get(ext, "audio/wav")
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def encode_audio_to_base64_ascii(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    ext = audio_path.lower().rsplit(".", 1)[-1]
    mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg", "aac": "audio/aac"}
    mime_type = mime_map.get(ext, "audio/wav")
    with open(audio_path, "rb") as f:
        ref_b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{ref_b64}"


def resolve_ref_audio(ref_audio, audio_encode: str = "utf-8") -> str | None:
    """Unified ref_audio handling shared by all models."""
    if ref_audio is None:
        return None
    if ref_audio.startswith(("http://", "https://", "data:")):
        return ref_audio
    if audio_encode == "utf-8":
        return encode_audio_to_base64_utf8(ref_audio)
    if audio_encode == "ascii":
        return encode_audio_to_base64_ascii(ref_audio)
    raise ValueError(f"Unknown audio_encode: {audio_encode}")


def resolve_ref_audio_list(ref_audio, audio_encode: str = "utf-8") -> str | list[str] | None:
    """Like :func:`resolve_ref_audio` but accepts a single value or a list."""
    if ref_audio is None:
        return None
    if isinstance(ref_audio, (list, tuple)):
        resolved: list[str] = []
        for item in ref_audio:
            value = resolve_ref_audio(item, audio_encode)
            if value is not None:
                resolved.append(value)
        return resolved[0] if len(resolved) == 1 else resolved
    return resolve_ref_audio(ref_audio, audio_encode)


def is_error_response(content: bytes) -> str | None:
    """Detect a JSON error response before treating bytes as audio."""
    try:
        text = content.decode("utf-8")
        if text.startswith('{"error"'):
            return text
    except UnicodeDecodeError:
        pass
    return None


def load_speaker_embedding(path: str, expected_dim: int = EXPECTED_SPEAKER_EMBEDDING_DIM) -> list[float]:
    """Load and validate a speaker embedding JSON file. Expected dimension can be overridden."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("speaker_embedding file must contain a JSON list")
    if len(data) != expected_dim:
        raise ValueError(
            f"Speaker embedding must have {expected_dim} values, got {len(data)}"
        )

    values = []
    for index, value in enumerate(data):
        try:
            values.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"speaker_embedding[{index}] must be a number, got {value!r}") from exc
    return values


def build_instruction_payload(args) -> str | None:
    """Return a string payload for the API `instructions` field."""
    instructions = getattr(args, "instructions", None)
    instruction_json = getattr(args, "instruction_json", None)
    if instructions and instruction_json:
        raise ValueError("Use either --instructions or --instruction-json, not both")
    if instruction_json:
        parsed = json.loads(instruction_json)
        return json.dumps(parsed, ensure_ascii=False)
    return instructions