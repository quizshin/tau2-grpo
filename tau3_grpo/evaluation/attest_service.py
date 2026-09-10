"""Write the checkpoint-to-vLLM service attestation immediately before exec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau3_grpo.evaluation.service_attestation import write_service_attestation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Attest a vLLM policy service launch")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    attestation = write_service_attestation(
        checkpoint_path=args.checkpoint,
        served_model_name=args.served_model_name,
        base_url=args.base_url,
        pid=args.pid,
        output=args.output,
    )
    print(json.dumps(attestation.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
