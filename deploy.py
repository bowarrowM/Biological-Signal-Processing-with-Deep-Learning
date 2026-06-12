"""
Deploys the Gradio app to Hugging Face Spaces.

Setup (one-time):
    huggingface-cli login    # uses your HF account token

Usage:
    python deploy.py                                  # repo name defaults to speech-emotion-recognition
    python deploy.py --name my-ser-demo               # custom repo name
    python deploy.py --public                         # make the Space public (default: public)
"""

import argparse
import os
import sys

from huggingface_hub import HfApi

SPACE_FILES = [
    "app.py",
    "data.py",
    "config.py",
    "config.yaml",
    "requirements.txt",
]

CHECKPOINT_FILES = [
    "checkpoints/best_model.keras",
    "checkpoints/label_encoder.pkl",
    "checkpoints/scaler.pkl",
]


def check_files():
    missing = [f for f in SPACE_FILES + CHECKPOINT_FILES if not os.path.exists(f)]
    if missing:
        print("Missing files — run training first:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",   default="speech-emotion-recognition", help="Space repo name")
    parser.add_argument("--public", action="store_true", default=True,    help="Make the Space public")
    args = parser.parse_args()

    check_files()

    api      = HfApi()
    username = api.whoami()["name"]
    repo_id  = f"{username}/{args.name}"

    print(f"Creating Space: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=not args.public,
        exist_ok=True,
    )

    for path in SPACE_FILES + CHECKPOINT_FILES:
        print(f"  Uploading {path}...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path,
            repo_id=repo_id,
            repo_type="space",
        )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nDone. Your Space will be live at:\n  {url}")
    print("\nBuilding takes ~2-3 minutes. Check the 'Logs' tab on HF if it doesn't start.")
