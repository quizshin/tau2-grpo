"""Write a run-specific simulator configuration, leaving shared defaults intact."""

import argparse
from pathlib import Path

import yaml

from tau3_grpo.models.compat import model_family
from tau3_grpo.paths import CONFIG_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--max-user-turns", type=int, default=15)
    parser.add_argument("--max-assistant-turns", type=int, default=15)
    parser.add_argument("--thinking", choices=["auto", "off", "on"], default="auto",
                        help="Simulator-only template setting; off also supports Qwen3.8 API aliases")
    args = parser.parse_args(argv)
    payload = yaml.safe_load((CONFIG_ROOT / "envs/interaction_config.yaml").read_text())
    config = payload["interaction"][0]["config"]
    config.update(
        user_model=args.model,
        user_base_url=args.base_url,
        max_user_turns=args.max_user_turns,
        max_assistant_turns=args.max_assistant_turns,
    )
    if args.thinking != "auto" or model_family(args.model) == "qwen35":
        config["user_llm_args"] = {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": args.thinking == "on"}}
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
