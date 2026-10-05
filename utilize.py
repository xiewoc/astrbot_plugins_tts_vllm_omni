import base64
import json
import os

import httpx

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


def read_embedding_json_text(source: str, timeout: float = 300.0) -> str:
    """Read speaker-embedding JSON text from a local file, URL or data URI.

    Args:
        source: Local JSON path, http(s) URL, or base64 ``data:`` URI.
        timeout: Download timeout in seconds for remote sources.

    Returns:
        The JSON text content.

    Raises:
        ValueError: The data URI is not base64 encoded or cannot be decoded.
        httpx.HTTPError: The remote source could not be fetched.
    """
    if source.startswith("data:"):
        header, _, payload = source.partition(",")
        if "base64" not in header:
            raise ValueError("speaker_embedding data URI must be base64 encoded")
        try:
            # binascii.Error / UnicodeDecodeError are both ValueError subclasses.
            return base64.b64decode(payload).decode("utf-8")
        except ValueError as exc:
            raise ValueError(f"speaker_embedding data URI decode failed: {exc}") from exc
    if source.startswith(("http://", "https://")):
        response = httpx.get(source, timeout=timeout)
        response.raise_for_status()
        return response.text
    with open(source, encoding="utf-8") as f:
        return f.read()


def load_speaker_embedding(
    source: str, expected_dim: int | None = EXPECTED_SPEAKER_EMBEDDING_DIM
) -> list[float]:
    """Load and validate a speaker embedding from a local file, URL or data URI.

    Args:
        source: Local JSON path, http(s) URL, or base64 ``data:`` URI.
        expected_dim: Expected embedding dimension; pass None to skip the check
            (Qwen3 and Ming use different dimensions).

    Returns:
        The embedding as a list of floats.

    Raises:
        ValueError: The payload is not a JSON list, the dimension mismatches,
            or an element is not a number.
    """
    data = json.loads(read_embedding_json_text(source))

    if not isinstance(data, list):
        raise ValueError("speaker_embedding file must contain a JSON list")
    if expected_dim is not None and len(data) != expected_dim:
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
