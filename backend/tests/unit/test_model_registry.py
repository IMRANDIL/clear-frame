from pathlib import Path

import pytest

from backend.app.domain.errors import ClearFrameError, ErrorCode
from backend.app.models.registry import load_model_spec, verify_model_weights


def test_baseline_model_manifest_is_complete() -> None:
    spec = load_model_spec("realesrgan-x4plus")

    assert spec.display_name == "RealESRGAN_x4plus"
    assert spec.architecture == "RRDBNet"
    assert spec.scale == 4
    assert not spec.temporal
    assert spec.release == "v0.1.0"
    assert spec.size_bytes == 67040989
    assert spec.sha256 == "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"


def test_unknown_model_is_reported_as_model_load_failure() -> None:
    with pytest.raises(ClearFrameError) as caught:
        load_model_spec("missing-model")

    assert caught.value.code is ErrorCode.MODEL_LOAD_FAILED


def test_weight_integrity_mismatch_is_rejected(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"not model weights")

    with pytest.raises(ClearFrameError) as caught:
        verify_model_weights(weights, load_model_spec("realesrgan-x4plus"))

    assert caught.value.code is ErrorCode.MODEL_LOAD_FAILED
    assert "mismatch" in str(caught.value)
