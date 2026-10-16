import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointdiff.runtime import make_dataset, read_config
from pointdiff.splits import verify_validation_split


def main():
    parser = argparse.ArgumentParser(description="Check annotation splits and feature coverage")
    parser.add_argument("--config", default="configs/charades_i3d.json")
    args = parser.parse_args()
    config = read_config(args.config)
    training = make_dataset(config, True)
    validation = make_dataset(config, False, split="validation")
    test = make_dataset(config, False, split="test")
    result = {"split": verify_validation_split(config, training, validation), "datasets": {}}
    for name, dataset in [("train", training), ("validation", validation), ("test", test)]:
        missing = sorted(
            {
                row["text_key"]
                for row in dataset.records
                if not dataset.text_store.contains(row["text_key"])
            }
        )
        if missing:
            raise FileNotFoundError(
                f"{name}: {len(missing)} query features missing; run tools/prepare_text.py"
            )
        sample = dataset[0]
        if sample["video"].shape[-1] != config["model"]["video_dim"]:
            raise ValueError(f"{name}: video feature dimension does not match config")
        if sample["text"].shape[-1] != config["model"]["text_dim"]:
            raise ValueError(f"{name}: text feature dimension does not match config")
        result["datasets"][name] = {
            "queries": len(dataset),
            "videos": len({row["video_id"] for row in dataset.records}),
            "video_dim": sample["video"].shape[-1],
            "text_dim": sample["text"].shape[-1],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
