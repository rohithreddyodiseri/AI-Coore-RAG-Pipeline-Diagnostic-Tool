import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset repository.")
    parser.add_argument("--repo-id", default="vectara/open_ragbench")
    parser.add_argument("--output", default="data/open_ragbench")
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} to {output_dir}...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        token=os.getenv("HF_TOKEN"),
    )
    print("Download complete.")


if __name__ == "__main__":
    main()
