import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointdiff.runtime import project_path, read_config
from pointdiff.text_features import prepare_distilbert_cache


def main():
    parser = argparse.ArgumentParser(description="Extract frozen DistilBERT query features")
    parser.add_argument("--config", default="configs/charades_i3d.json")
    parser.add_argument(
        "--model",
        default="distilbert/distilbert-base-uncased",
        help="Hugging Face repository or local pretrained model directory",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    config = read_config(args.config)
    model = project_path(args.model)
    if not model.is_dir():
        from huggingface_hub import snapshot_download

        model = Path(
            snapshot_download(
                repo_id=args.model,
                revision=args.revision,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "tokenizer_config.json",
                    "tokenizer.json",
                    "vocab.txt",
                    "special_tokens_map.json",
                ],
            )
        )
    data = config["data"]
    annotations = [
        project_path(data[key])
        for key in ["train_annotations", "validation_annotations", "test_annotations"]
    ]
    prepare_distilbert_cache(
        annotations,
        model,
        project_path(data["text_features"]),
        max_query_len=data.get("max_query_len", 32),
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
