import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_config(path):
    with project_path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def configure_runtime(seed, threads, device):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; use --device cpu or a CUDA-enabled environment"
            )
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device(device)


def rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def model_inputs(batch, device, training=False):
    keys = ["video", "text", "video_mask", "text_mask"]
    if training:
        keys += ["point", "query_id", "gpc_video", "gpc_video_mask"]
    return {
        key: batch[key].to(device) if torch.is_tensor(batch[key]) else batch[key] for key in keys
    }


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def source_fingerprints():
    paths = list((ROOT / "pointdiff").rglob("*.py"))
    paths += [ROOT / "train.py", ROOT / "inference.py"]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def implementation_info(model):
    return {
        "gpc_grid": "native",
        "gpc_feature_source": model.gpc_feature_source,
        "gpc_weight_threshold": model.gpc.weight_threshold,
        "gaussian_coordinates": model.gpc.gaussian_coordinates,
        "gpc_boundary_mode": model.gpc.boundary_mode,
        "gpc_device": model.gpc_device,
        "condition_prior": model.condition_prior,
        "matching_method": model.matching_method,
        "noise_loss_mode": model.noise_loss_mode,
        "sampling_method": model.sampling_method,
        "auxiliary_loss": model.auxiliary_loss,
        "auxiliary_loss_weight": model.auxiliary_loss_weight,
    }


def inference_protocol(model):
    return "Video and query inputs only; Gaussian initialization and x0-derived DDIM sampling."


def make_dataset(config, training, limit=None, split="eval"):
    from .data import PointDiffDataset

    data = config["data"]
    if training:
        annotation_key = "train_annotations"
    elif split == "validation":
        annotation_key = "validation_annotations"
    elif split == "test":
        annotation_key = "test_annotations"
    elif split == "eval":
        annotation_key = "eval_annotations"
    else:
        raise ValueError("Unknown split: " + split)
    return PointDiffDataset(
        annotation_file=project_path(data[annotation_key]),
        video_features=project_path(data["video_features"]),
        text_features=project_path(data["text_features"]),
        dataset_name=data["dataset_name"],
        training=training,
        num_clips=data["num_clips"],
        max_query_len=data.get("max_query_len", 32),
        cache_video=data.get("cache_video", True),
        limit=limit,
        video_cache_capacity=data.get("video_cache_capacity", 512),
        cache_text=data.get("cache_text", False),
        text_cache_capacity=data.get("text_cache_capacity"),
        include_gpc_video=training,
        gpc_video_cache_capacity=data.get("gpc_video_cache_capacity", 128),
    )
