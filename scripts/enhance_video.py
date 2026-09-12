from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import PipelineConfig, parse_tile_size  # noqa: E402
from backend.app.domain.errors import ClearFrameError, ErrorCode  # noqa: E402
from backend.app.observability.logging import configure_logging, log_event  # noqa: E402
from backend.app.pipeline.pipeline import run_pipeline  # noqa: E402


def _tile_argument(value: str) -> int | None:
    try:
        return parse_tile_size(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser(defaults: PipelineConfig | None = None) -> argparse.ArgumentParser:
    config = defaults or PipelineConfig.from_environment()
    parser = argparse.ArgumentParser(
        description="Restore a low-quality video to an aspect-preserving 1080p MP4.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Source MP4, MOV, or WebM")
    parser.add_argument("--output", required=True, type=Path, help="Destination MP4")
    parser.add_argument("--report", type=Path, help="Optional processing report path")
    parser.add_argument("--model", default=config.model_name, help="Registered restoration model")
    parser.add_argument("--model-dir", type=Path, default=config.model_directory)
    parser.add_argument("--temp-dir", type=Path, default=config.temporary_directory)
    parser.add_argument("--device", default=config.device, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--tile",
        type=_tile_argument,
        default=config.tile_size,
        metavar="auto|PIXELS",
        help="Inference tile size; default: auto",
    )
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--crf", type=int, default=config.crf)
    parser.add_argument("--preset", default=config.encoder_preset)
    parser.add_argument("--aac-bitrate-kbps", type=int, default=config.aac_bitrate_kbps)
    parser.add_argument("--ffmpeg", default=config.ffmpeg_path)
    parser.add_argument("--ffprobe", default=config.ffprobe_path)
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 CUDA inference")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _pipeline_config(arguments: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        ffmpeg_path=arguments.ffmpeg,
        ffprobe_path=arguments.ffprobe,
        model_name=arguments.model,
        model_directory=arguments.model_dir,
        temporary_directory=arguments.temp_dir,
        device=arguments.device,
        tile_size=arguments.tile,
        batch_size=arguments.batch_size,
        use_half=False if arguments.fp32 else None,
        crf=arguments.crf,
        encoder_preset=arguments.preset,
        aac_bitrate_kbps=arguments.aac_bitrate_kbps,
        overwrite=arguments.overwrite,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    logger = configure_logging(verbose=arguments.verbose)

    def observe(event: str, fields: Mapping[str, object]) -> None:
        log_event(logger, event, **fields)

    try:
        result = run_pipeline(
            arguments.input,
            arguments.output,
            config=_pipeline_config(arguments),
            report_path=arguments.report,
            observer=observe,
        )
    except KeyboardInterrupt:
        log_event(logger, "pipeline_cancelled", error_code=ErrorCode.CANCELLED)
        return 130
    except ClearFrameError as exc:
        log_event(
            logger,
            "pipeline_failed",
            error_code=exc.code,
            error_message=str(exc),
            retryable=exc.retryable,
        )
        return 1
    except (OSError, ValueError) as exc:
        log_event(
            logger,
            "pipeline_failed",
            error_code="CONFIGURATION_ERROR",
            error_message=str(exc),
            retryable=False,
        )
        return 2
    except Exception:
        logger.exception(
            "pipeline_failed",
            extra={"clearframe_fields": {"error_code": "UNEXPECTED_ERROR"}},
        )
        return 1

    log_event(
        logger,
        "result_ready",
        output=str(result.output_path),
        report=str(result.report_path),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
