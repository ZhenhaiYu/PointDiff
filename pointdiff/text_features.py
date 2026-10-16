import hashlib
import json
import os
from pathlib import Path


def read_queries(annotation_files, limit=None):
    unique = {}
    for annotation_file in annotation_files:
        with open(annotation_file, encoding="utf-8") as handle:
            annotations = json.load(handle)
        count = 0
        for annotation in annotations.values():
            for query in annotation["sentences"]:
                if limit is not None and count >= limit:
                    break
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("Invalid empty query in %s" % annotation_file)
                key = hashlib.sha256(query.encode("utf-8")).hexdigest()
                unique.setdefault(key, query)
                count += 1
            if limit is not None and count >= limit:
                break
    return unique


def checkpoint_fingerprint(model_path):
    model_path = Path(model_path)
    weight_names = []
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (model_path / name).is_file():
            weight_names.append(name)
            break
    if not weight_names:
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if (model_path / name).is_file():
                index = json.loads((model_path / name).read_text())
                weight_names = sorted(set(index["weight_map"].values()))
                break
    if not weight_names:
        raise FileNotFoundError(
            "No pretrained DistilBERT weights in %s; random text is unsupported" % model_path
        )
    digest = hashlib.sha256()
    for name in ["config.json"] + weight_names:
        digest.update(name.encode("utf-8"))
        with open(model_path / name, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_distilbert_cache(
    annotation_files,
    model_path,
    output_dir,
    max_query_len=32,
    batch_size=64,
    device="cpu",
    limit=None,
):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, DistilBertModel

    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(
            "--text-model must be a complete LOCAL pretrained DistilBERT folder"
        )
    if max_query_len <= 0 or batch_size <= 0:
        raise ValueError("max_query_len and batch_size must be positive")
    source_fingerprint = checkpoint_fingerprint(model_path)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    expected = {
        "format_version": 1,
        "encoder": "distilbert-base-uncased",
        "embedding": "last_hidden_state",
        "dimension": 768,
        "max_query_len": int(max_query_len),
        "special_tokens_removed": True,
        "storage_dtype": "float16",
        "query_key": "sha256_exact_utf8_query",
        "pretrained_checkpoint_sha256": source_fingerprint,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(
                    "Cache provenance mismatch for %s; use a new text-output directory" % key
                )
    elif next(output_dir.glob("*.npy"), None) is not None:
        raise ValueError(
            "Existing query features lack provenance; use an empty text-output directory"
        )
    queries = read_queries(annotation_files, limit)
    pending = [
        (key, query)
        for (key, query) in queries.items()
        if not (output_dir / (key + ".npy")).exists()
    ]
    manifest = dict(
        expected,
        source_model=str(model_path),
        annotation_files=[str(Path(path).resolve()) for path in annotation_files],
        requested_unique_queries=len(queries),
        cached_queries=len(queries) - len(pending),
    )
    atomic_json(manifest_path, manifest)
    if not pending:
        print("All %d requested DistilBERT queries are already cached." % len(queries), flush=True)
        return manifest
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
    model = DistilBertModel.from_pretrained(str(model_path), local_files_only=True)
    if model.config.model_type != "distilbert" or model.config.dim != 768:
        raise ValueError("Expected pretrained DistilBERT base (768-D)")
    if max_query_len + 2 > model.config.max_position_embeddings:
        raise ValueError("Requested query length exceeds DistilBERT's positional limit")
    model.eval().requires_grad_(False).to(device)
    completed = len(queries) - len(pending)
    with torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            examples = pending[start : start + batch_size]
            encoded = tokenizer(
                [query for (_, query) in examples],
                padding=True,
                truncation=True,
                max_length=max_query_len + 2,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            special_mask = encoded.pop("special_tokens_mask").bool()
            valid_mask = encoded["attention_mask"].bool() & ~special_mask
            hidden = model(
                **{key: value.to(device) for (key, value) in encoded.items()}
            ).last_hidden_state.cpu()
            for index, (key, _) in enumerate(examples):
                features = hidden[index, valid_mask[index]].numpy().astype(np.float16)
                if features.shape[0] == 0 or not np.isfinite(features).all():
                    raise ValueError("Empty/nonfinite DistilBERT features for %s" % key)
                destination = output_dir / (key + ".npy")
                temporary = output_dir / (key + ".tmp.%d" % os.getpid())
                with open(temporary, "wb") as handle:
                    np.save(handle, features, allow_pickle=False)
                os.replace(temporary, destination)
            completed += len(examples)
            manifest["cached_queries"] = completed
            atomic_json(manifest_path, manifest)
            if start == 0 or completed == len(queries) or start // batch_size % 10 == 0:
                print("DistilBERT cache: %d/%d queries" % (completed, len(queries)), flush=True)
    return manifest
