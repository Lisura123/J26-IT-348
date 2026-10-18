"""Regenerate firmware/pv_fault/selfcheck/pv_selfcheck.h from the frozen model.

Usage: python scripts/09_generate_selfcheck.py [--check]

Reads only models/model_contract.json and models/int8/quant_params.json (both
verified against their pinned SHA-256 values first); no dataset is used.
--check compares instead of writing (exit 1 if the committed file differs).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinyml.export import file_sha256  # noqa: E402
from tinyml.holdout import verify_frozen_model  # noqa: E402
from tinyml.quantize import QuantParams  # noqa: E402
from tinyml.selfcheck import HEADER, selfcheck_header  # noqa: E402

PINNED_QUANT_PARAMS = "5daa66d21098b20845ea7b9d9d891a7a43fb62fd4d225b9b348936f5de62ab25"
OUT = ROOT / "firmware" / "pv_fault" / "selfcheck" / HEADER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    contract = verify_frozen_model(ROOT / "models")
    qp = ROOT / "models" / "int8" / "quant_params.json"
    if file_sha256(qp) != PINNED_QUANT_PARAMS:
        sys.exit("quant_params.json is not the frozen int8-v1 file")
    text = selfcheck_header(contract, file_sha256(ROOT / "models" / "model_contract.json"),
                            QuantParams.from_json(qp.read_text(encoding="utf-8")))
    if a.check:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print("pv_selfcheck.h up to date" if same else "pv_selfcheck.h differs")
        sys.exit(0 if same else 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(json.dumps({"written": str(OUT.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
