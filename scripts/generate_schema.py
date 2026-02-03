from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tgefd.config.models import TGEFDConfig


def main() -> int:
    schema = TGEFDConfig.model_json_schema()
    out_dir = Path("schemas")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tgefd_config_v1.json"
    out_path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    print(f"JSON Schema generated: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
