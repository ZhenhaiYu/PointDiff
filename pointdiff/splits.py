import hashlib
import json
from pathlib import Path
from .runtime import project_path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_validation_split(config, training, validation):
    train_ids = {r["video_id"] for r in training.records}
    val_ids = {r["video_id"] for r in validation.records}
    if train_ids & val_ids:
        raise ValueError("Training and validation must have disjoint videos")
    data = config["data"]
    test_path = data.get("test_annotations")
    if test_path:
        test_ids = set(json.loads(project_path(test_path).read_text()))
        if (train_ids | val_ids) & test_ids:
            raise ValueError("Training/validation overlap the official test set")
    if data.get("split_manifest"):
        manifest = json.loads(project_path(data["split_manifest"]).read_text())
        for key, hash_key in [
            ("train_annotations", "train_annotations_sha256"),
            ("validation_annotations", "validation_annotations_sha256"),
        ]:
            if sha256_file(project_path(data[key])) != manifest[hash_key]:
                raise ValueError("Annotation file differs from split manifest: " + key)
        if not train_ids <= set(manifest["train_videos"]) or not val_ids <= set(
            manifest["validation_videos"]
        ):
            raise ValueError("Dataset videos differ from split manifest")
    return {
        "train_videos": len(train_ids),
        "validation_videos": len(val_ids),
        "train_queries": len(training),
        "validation_queries": len(validation),
        "video_overlap": 0,
    }
