"""Run the local dashboard with ``python -m biosqi.server``."""

from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("biosqi.server.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
