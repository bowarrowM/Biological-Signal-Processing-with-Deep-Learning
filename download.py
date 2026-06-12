"""
Downloads all datasets from Kaggle to data/raw/.

Setup (one-time):
    pip install kaggle
    # Go to kaggle.com → Account → Create API Token → save as ~/.kaggle/kaggle.json

Usage:
    python download.py                      # downloads to data/raw/
    python download.py --output /my/data    # custom output directory
    python download.py --dataset tess       # download one dataset only
"""

import argparse
import os

import kaggle

# Kaggle dataset slugs — verify at kaggle.com/datasets if a download fails
DATASETS = {
    "ravdess": "uwrfkaggler/ravdess-emotional-speech-audio",
    "crema":   "ejlok1/cremad",
    "tess":    "ejlok1/toronto-emotional-speech-set-tess",
    "savee":   "ejlok1/surrey-audiovisual-expressed-emotion-savee",
}


def download(name: str, slug: str, output_dir: str) -> str:
    dest = os.path.join(output_dir, name)
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {name}  ({slug})\n  → {os.path.abspath(dest)}")
    kaggle.api.dataset_download_files(slug, path=dest, unzip=True, quiet=False)
    print(f"  Done.\n")
    return os.path.abspath(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  default="data/raw", help="Root directory for downloaded datasets")
    parser.add_argument("--dataset", default=None, choices=list(DATASETS), help="Download one dataset only")
    args = parser.parse_args()

    kaggle.api.authenticate()

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    paths   = {}

    for name, slug in targets.items():
        paths[name] = download(name, slug, args.output)

    print("Finished. Add these paths to config.yaml or export as environment variables:\n")
    env_map = {"ravdess": "RAVDESS_PATH", "crema": "CREMA_PATH",
               "tess": "TESS_PATH", "savee": "SAVEE_PATH"}
    for name, path in paths.items():
        print(f"  export {env_map[name]}={path}")
