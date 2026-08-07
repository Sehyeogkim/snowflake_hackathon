from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx

from mavis.backends.openai import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_TERRA_MODEL,
    OpenAIInferenceBackend,
    TieredOpenAIInferenceBackend,
)
from mavis.models import CandidateFrame, CandidateSet, ClipRecord

LABELS = ("safe_walkway", "safe_walkway_violation")


def _candidates(tmp_path: Path) -> tuple[CandidateSet, list[bytes]]:
    contents = [b"early-image", b"peak-image", b"late-image"]
    paths = [tmp_path / f"{name}.jpg" for name in ("early", "peak", "late")]
    for path, content in zip(paths, contents, strict=True):
        path.write_bytes(content)
    frames = [
        CandidateFrame(name=path.stem, frame_index=index, timestamp_s=float(index), path=path)
        for index, path in enumerate(paths)
    ]
    return (
        CandidateSet(
            clip_id="clip-1",
            fps=1.0,
            frame_count=3,
            duration_s=3.0,
            early=frames[0],
            peak=frames[1],
            late=frames[2],
        ),
        contents,
    )


def _response_payload() -> dict[str, object]:
    structured = {
        "prediction": "safe_walkway_violation",
        "scores": {"safe_walkway": 0.2, "safe_walkway_violation": 0.8},
        "scene": "worker crossing outside the marked route",
        "risk": 0.3,
        "need_temporal_context": False,
    }
    return {
        "id": "resp_test_123",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(structured)}],
            }
        ],
        "usage": {"input_tokens": 101, "output_tokens": 29, "total_tokens": 130},
    }


def test_infer_sends_vision_schema_and_preserves_usage(tmp_path: Path) -> None:
    candidates, expected_images = _candidates(tmp_path)
    secret = "test-key-never-log"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.openai.com/v1/responses"
        assert request.headers["Authorization"] == f"Bearer {secret}"
        payload = json.loads(request.content)
        assert payload["model"] == DEFAULT_OPENAI_MODEL
        assert payload["store"] is False
        assert payload["reasoning"] == {"effort": "none"}

        schema_format = payload["text"]["format"]
        assert schema_format["type"] == "json_schema"
        assert schema_format["strict"] is True
        assert schema_format["schema"]["properties"]["prediction"]["enum"] == list(LABELS)
        assert schema_format["schema"]["properties"]["scores"]["required"] == list(LABELS)

        content = payload["input"][0]["content"]
        assert content[0]["type"] == "input_text"
        assert "chronological" in content[0]["text"]
        image_parts = content[1:]
        assert len(image_parts) == 3
        for part, expected in zip(image_parts, expected_images, strict=True):
            assert part["type"] == "input_image"
            assert part["detail"] == "low"
            prefix, encoded = part["image_url"].split(",", 1)
            assert prefix == "data:image/jpeg;base64"
            assert base64.b64decode(encoded) == expected
        return httpx.Response(200, json=_response_payload())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OpenAIInferenceBackend(labels=LABELS, api_key=secret, client=client)
    result = backend.infer(
        ClipRecord(clip_id="clip-1", path=tmp_path / "clip.mp4", label=LABELS[1]),
        candidates,
        "strong_multi",
    )
    backend.close()

    assert result.error is None
    assert result.model == DEFAULT_OPENAI_MODEL
    assert result.prediction == "safe_walkway_violation"
    assert result.query_id == "resp_test_123"
    assert result.risk == 0.8
    assert result.raw_response["usage"] == {
        "input_tokens": 101,
        "output_tokens": 29,
        "total_tokens": 130,
    }
    assert result.raw_response["parsed"]["scene"] == result.scene
    assert not client.is_closed
    client.close()


def test_http_failure_does_not_expose_api_key(tmp_path: Path) -> None:
    candidates, _ = _candidates(tmp_path)
    secret = "super-secret-openai-key"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {secret}"
        return httpx.Response(401, json={"error": {"message": "invalid authentication"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OpenAIInferenceBackend(labels=LABELS, api_key=secret, client=client)
    result = backend.infer(
        ClipRecord(clip_id="clip-1", path=tmp_path / "clip.mp4", label=LABELS[0]),
        candidates,
        "cheap_single",
    )

    assert result.error == "HTTPStatusError: OpenAI Responses API returned HTTP 401"
    assert secret not in result.error
    assert secret not in repr(result.raw_response)
    client.close()


def test_infer_paths_retries_and_keeps_arbitrary_chronological_order(tmp_path: Path) -> None:
    paths = [tmp_path / f"frame-{index}.png" for index in range(4)]
    expected_images = [f"image-{index}".encode() for index in range(4)]
    for path, content in zip(paths, expected_images, strict=True):
        path.write_bytes(content)

    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        payload = json.loads(request.content)
        image_parts = payload["input"][0]["content"][1:]
        assert len(image_parts) == 4
        observed = [base64.b64decode(part["image_url"].split(",", 1)[1]) for part in image_parts]
        assert observed == expected_images
        if attempts == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        if attempts == 2:
            return httpx.Response(503, json={"error": {"message": "temporarily unavailable"}})
        return httpx.Response(200, json=_response_payload())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OpenAIInferenceBackend(
        labels=LABELS,
        api_key="test-key",
        client=client,
        retry_base_seconds=0.75,
        retry_max_seconds=1.0,
        sleep=delays.append,
    )

    result = backend.infer_paths(paths, "chronological_review")

    assert result.error is None
    assert attempts == 3
    assert delays == [0.75, 1.0]
    client.close()


def test_retry_stops_at_configured_max_attempts(tmp_path: Path) -> None:
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"image")
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, json={"error": {"message": "server error"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OpenAIInferenceBackend(
        labels=LABELS,
        api_key="test-key",
        client=client,
        max_attempts=3,
        retry_base_seconds=0.1,
        sleep=delays.append,
    )

    result = backend.infer_paths([path], "single_review")

    assert result.error == "HTTPStatusError: OpenAI Responses API returned HTTP 500"
    assert attempts == 3
    assert delays == [0.1, 0.2]
    client.close()


def test_tiered_backend_routes_actions_to_model_and_detail(tmp_path: Path) -> None:
    candidates, _ = _candidates(tmp_path)
    observed: list[tuple[str, list[str]]] = []
    secret = "shared-tier-key"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {secret}"
        payload = json.loads(request.content)
        image_parts = payload["input"][0]["content"][1:]
        observed.append((payload["model"], [part["detail"] for part in image_parts]))
        return httpx.Response(200, json=_response_payload())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = TieredOpenAIInferenceBackend(labels=LABELS, api_key=secret, client=client)
    clip = ClipRecord(clip_id="clip-1", path=tmp_path / "clip.mp4", label=LABELS[0])

    cheap_result = backend.infer(clip, candidates, "cheap_single")
    strong_single_result = backend.infer(clip, candidates, "strong_single")
    strong_multi_result = backend.infer(clip, candidates, "strong_multi")
    backend.close()

    assert cheap_result.model == DEFAULT_OPENAI_MODEL
    assert strong_single_result.model == DEFAULT_OPENAI_TERRA_MODEL
    assert strong_multi_result.model == DEFAULT_OPENAI_TERRA_MODEL
    assert observed == [
        (DEFAULT_OPENAI_MODEL, ["low"]),
        (DEFAULT_OPENAI_TERRA_MODEL, ["original"]),
        (DEFAULT_OPENAI_TERRA_MODEL, ["original", "original", "original"]),
    ]
    assert backend.evidence_source == "openai"
    assert not client.is_closed
    client.close()


def test_tiered_infer_paths_supports_explicit_tier(tmp_path: Path) -> None:
    paths = [tmp_path / f"frame-{index}.jpg" for index in range(2)]
    for index, path in enumerate(paths):
        path.write_bytes(f"frame-{index}".encode())
    observed: list[tuple[str, list[str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        image_parts = payload["input"][0]["content"][1:]
        observed.append((payload["model"], [part["detail"] for part in image_parts]))
        return httpx.Response(200, json=_response_payload())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = TieredOpenAIInferenceBackend(labels=LABELS, api_key="test-key", client=client)

    result = backend.infer_paths(paths, "custom_temporal_review", tier="terra")

    assert result.error is None
    assert result.model == DEFAULT_OPENAI_TERRA_MODEL
    assert observed == [(DEFAULT_OPENAI_TERRA_MODEL, ["original", "original"])]
    client.close()
