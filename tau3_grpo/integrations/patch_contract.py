"""Contract checks for the minimal local patches applied to veRL v0.7.1.

Milestone D5. The patches are deliberately small and every one carries the marker
comment ``Tau3-GRPO local patch``. These checks are assertions about the vendored
tree, so a veRL upgrade that silently drops a patch fails a CPU test instead of
producing quietly wrong advantages.

The four patch sites:

1. ``verl/experimental/agent_loop/tool_agent_loop.py`` — emit per-generation
   anchor id / token span, ``None`` at tool and user observation segments.
2. ``verl/trainer/ppo/ray_trainer.py::compute_advantage`` — pass
   ``non_tensor_batch`` to the ``tau_gigpo`` estimator.
3. ``verl/trainer/ppo/ray_trainer.py::fit`` — apply the Dynamic Filtering response
   mask before ``compute_advantage``.
4. ``verl/trainer/config/algorithm.py::AlgoConfig`` — accept ``dynamic_filter``
   and ``gigpo`` blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tau3_grpo.paths import VERL_ROOT

PATCH_MARKER = "Tau3-GRPO local patch"

TOOL_AGENT_LOOP = Path("verl/experimental/agent_loop/tool_agent_loop.py")
RAY_TRAINER = Path("verl/trainer/ppo/ray_trainer.py")
ALGORITHM_CONFIG = Path("verl/trainer/config/algorithm.py")


@dataclass(frozen=True)
class PatchRequirement:
    """One patched file and the tokens that must appear in it."""

    relative_path: Path
    required_tokens: tuple[str, ...]
    description: str


REQUIREMENTS: tuple[PatchRequirement, ...] = (
    PatchRequirement(
        relative_path=Path("setup.py"),
        required_tokens=(PATCH_MARKER, "QWEN35_REQUIRES", '"numpy<2"', '"vllm==0.20.0"'),
        description="Qwen3.5 dependencies are opt-in; the legacy vLLM extra retains NumPy 1",
    ),
    PatchRequirement(
        relative_path=Path("verl/models/transformers/monkey_patch.py"),
        required_tokens=(PATCH_MARKER, 'model.config.model_type == "qwen3_5"',
                         "Tau3 Qwen3.5 requires padded native forward"),
        description="Qwen3.5 rejects generic packing/sequence parallel patches",
    ),
    PatchRequirement(
        relative_path=Path("verl/utils/tokenizer.py"),
        required_tokens=(PATCH_MARKER, "TAU3_GRPO_TEXT_ONLY", 'config.model_type == "qwen3_5"'),
        description="Explicit text-only Qwen3.5 launchers skip multimodal processing",
    ),
    PatchRequirement(
        relative_path=TOOL_AGENT_LOOP,
        required_tokens=(
            PATCH_MARKER,
            "anchor_ids",
            "anchor_spans",
            "tau3_anchor_hook",
            "TAU3_GRPO_ANCHOR_HOOK",
            "finalize_rollout",
            "reward_score=terminal_reward_score",
        ),
        description=(
            "ToolAgentLoop emits aligned anchors, loads the worker hook, and "
            "publishes the official terminal reward"
        ),
    ),
    PatchRequirement(
        relative_path=RAY_TRAINER,
        required_tokens=(
            PATCH_MARKER,
            "tau_gigpo",
            "register_tau3_gigpo",
            "non_tensor_batch",
            "tau3_dynamic_filter",
            "tau3_pad_policy_batch",
            "tau3_unpad_policy_batch",
            "TAU3_GRPO_POLICY_BATCH_DIVISOR",
        ),
        description=(
            "ray_trainer threads non_tensor_batch, applies DF, and masks exact-DP "
            "dummy padding"
        ),
    ),
    PatchRequirement(
        relative_path=ALGORITHM_CONFIG,
        required_tokens=(PATCH_MARKER, "dynamic_filter", "gigpo"),
        description="AlgoConfig accepts dynamic_filter and gigpo blocks",
    ),
)


def verl_root() -> Path:
    return VERL_ROOT


def check_patch(requirement: PatchRequirement, *, root: Path | None = None) -> list[str]:
    """Return a list of problems for one requirement; empty means satisfied."""

    base = root or verl_root()
    path = base / requirement.relative_path
    if not path.is_file():
        return [f"missing patched file: {path}"]
    text = path.read_text(encoding="utf-8")
    return [
        f"{requirement.relative_path}: missing token {token!r} ({requirement.description})"
        for token in requirement.required_tokens
        if token not in text
    ]


def verify_all(root: Path | None = None) -> list[str]:
    """Return every unsatisfied patch contract."""

    problems: list[str] = []
    for requirement in REQUIREMENTS:
        problems.extend(check_patch(requirement, root=root))
    return problems


def assert_patched(root: Path | None = None) -> None:
    problems = verify_all(root)
    if problems:
        raise RuntimeError("veRL local patches are missing or incomplete:\n" + "\n".join(problems))


def patched_files(root: Path | None = None) -> list[Path]:
    base = root or verl_root()
    return [base / requirement.relative_path for requirement in REQUIREMENTS]


def vanilla_grpo_untouched(root: Path | None = None) -> bool:
    """Confirm the GRPO branch of `compute_advantage` still exists verbatim.

    The patch must not change the default path, so this looks for the original
    call that vanilla GRPO takes.
    """

    base = root or verl_root()
    text = (base / RAY_TRAINER).read_text(encoding="utf-8")
    return "core_algos.compute_grpo_outcome_advantage(" in text
