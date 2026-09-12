import pytest

from backend.app.pipeline.decoder import DecoderConfig


@pytest.mark.parametrize("batch_size", [0, 9])
def test_decoder_batch_size_is_bounded(batch_size: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 8"):
        DecoderConfig(batch_size=batch_size)
