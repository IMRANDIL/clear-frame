from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ClearFrame local web application.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    arguments = parser.parse_args(argv)
    uvicorn.run(
        "backend.app.web.api:app",
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
