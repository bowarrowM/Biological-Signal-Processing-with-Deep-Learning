"""
Loads config.yaml and applies overrides from CLI args and environment variables.

Priority (highest to lowest):
  1. CLI argument  (--ravdess-path, --crema-path, ...)
  2. Environment variable  (RAVDESS_PATH, CREMA_PATH, ...)
  3. config.yaml value

Usage:
  from config import load_config
  cfg = load_config()          # uses sys.argv
  cfg = load_config([])        # no CLI args (e.g. in a notebook)
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

CONFIG_FILE = Path(__file__).parent / "config.yaml"

# Maps config.yaml key path → (CLI flag, env var name)
_PATH_OVERRIDES = {
    ("data", "ravdess_path"): ("--ravdess-path", "RAVDESS_PATH"),
    ("data", "crema_path"):   ("--crema-path",   "CREMA_PATH"),
    ("data", "tess_path"):    ("--tess-path",    "TESS_PATH"),
    ("data", "savee_path"):   ("--savee-path",   "SAVEE_PATH"),
}


def _load_yaml() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition — training / inference",
        add_help=True,
    )
    parser.add_argument("--config", default=None, help="Path to an alternate config.yaml")
    for (_, key), (flag, _) in _PATH_OVERRIDES.items():
        parser.add_argument(flag, default=None, dest=key, help=f"Override config.yaml data.{key}")
    return parser


def load_config(argv=None) -> dict:
    """Return the merged config dict."""
    parser = _build_parser()
    # parse_known_args so callers (e.g. pytest, Jupyter) don't choke on unknown flags
    args, _ = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    cfg_path = Path(args.config) if args.config else CONFIG_FILE
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    for (section, key), (_, env_var) in _PATH_OVERRIDES.items():
        # CLI arg takes priority, then env var, then yaml value
        cli_val = getattr(args, key, None)
        env_val = os.environ.get(env_var)
        if cli_val:
            cfg[section][key] = cli_val
        elif env_val:
            cfg[section][key] = env_val

    return cfg


def require_paths(cfg: dict, *keys: str) -> None:
    """Raise a clear error if any dataset path is missing."""
    missing = []
    for key in keys:
        val = cfg["data"].get(key, "")
        if not val:
            _, env_var = next(v for (_, k), v in _PATH_OVERRIDES.items() if k == key)
            flag = next(f for (_, k), (f, _) in _PATH_OVERRIDES.items() if k == key)
            missing.append(f"  {key}: set via {flag} or ${env_var}")
    if missing:
        raise SystemExit(
            "Missing required dataset paths:\n" + "\n".join(missing)
        )
