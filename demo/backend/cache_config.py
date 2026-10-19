"""Locate the four cached feature-statistics files without importing GPU libraries."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_NAMES = ("sentiment", "toxicity", "debate", "polite")


def cache_directory():
    """Resolve relative overrides against the repository, not the launch directory."""
    configured_path = Path(os.getenv("NEUROSTEER_CACHE_DIR", "results/demo/demo_v1")).expanduser()
    return configured_path if configured_path.is_absolute() else PROJECT_ROOT / configured_path


def discover_caches(model_name="gpt2-small", layer=6):
    """Require one matching cache per task; ignore unrelated runs and hidden files."""
    directory = cache_directory()
    task_paths = {}
    for task_name in TASK_NAMES:
        matching_paths = sorted(directory.glob(
            f"{model_name}_{task_name}_layer_{layer}_*_topK_*/*neuron_info*.pkl"
        ))
        if len(matching_paths) != 1:
            raise ValueError(
                f"Expected one {task_name} cache for {model_name}, layer {layer}; "
                f"found {len(matching_paths)} under {directory}. "
                "Set NEUROSTEER_CACHE_DIR to your prepared cache directory."
            )
        task_paths[task_name] = matching_paths[0]
    return task_paths
