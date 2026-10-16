import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

DATASET_ALIASES = {
    "charades": "charades",
    "charades_sta": "charades",
    "charades-sta": "charades",
    "activitynet": "activitynet",
    "activitynet_captions": "activitynet",
    "activitynet-captions": "activitynet",
    "tacos": "tacos",
}


def canonical_dataset_name(name):
    try:
        return DATASET_ALIASES[str(name).lower()]
    except KeyError as exc:
        raise ValueError("Expected charades, activitynet, or tacos; got %r" % name) from exc


def query_cache_key(query):
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def load_pointdiff_annotations(annotation_file, dataset_name, training):
    dataset_name = canonical_dataset_name(dataset_name)
    with open(annotation_file, encoding="utf-8") as handle:
        annotations = json.load(handle)
    records = []
    for video_id, annotation in annotations.items():
        fps = float(annotation["fps"]) if dataset_name == "tacos" else 1.0
        duration = (
            float(annotation["num_frames"]) / fps
            if dataset_name == "tacos"
            else float(annotation["duration"])
        )
        if not math.isfinite(duration) or duration <= 0 or fps <= 0:
            raise ValueError("Invalid duration/fps for %s" % video_id)
        sentences = annotation["sentences"]
        if training:
            labels = annotation["glance"] if dataset_name == "tacos" else annotation["glances"]
            if len(labels) != len(sentences):
                raise ValueError("Point/query count mismatch for %s" % video_id)
        else:
            labels = annotation["timestamps"]
            if len(labels) != len(sentences):
                raise ValueError("Interval/query count mismatch for %s" % video_id)
        for index, (sentence, label) in enumerate(zip(sentences, labels)):
            if not isinstance(sentence, str) or not sentence.strip():
                raise ValueError("Empty or invalid sentence for %s:%d" % (video_id, index))
            record = {
                "video_id": video_id,
                "query": sentence,
                "duration": duration,
                "query_id": "%s:%s:%d" % (dataset_name, video_id, index),
                "text_key": query_cache_key(sentence),
            }
            if training:
                point_seconds = float(label) / fps
                if not math.isfinite(point_seconds) or point_seconds < 0:
                    raise ValueError("Invalid supplied point for %s" % record["query_id"])
                record["point_seconds"] = point_seconds
                record["point"] = point_seconds / duration
            else:
                (start, end) = (float(label[0]) / fps, float(label[1]) / fps)
                if not math.isfinite(start) or not math.isfinite(end):
                    raise ValueError("Nonfinite evaluation interval for %s" % record["query_id"])
                (start, end) = (max(0.0, start), min(duration, end))
                if end <= start:
                    continue
                record["target"] = [start, end]
            records.append(record)
    return records


def _video_keys(video_id):
    keys = [video_id, Path(video_id).stem]
    if video_id.startswith("v_"):
        keys.append(video_id[2:])
    else:
        keys.append("v_" + video_id)
    return list(dict.fromkeys(keys))


def _feature_array(array, name):
    array = np.asarray(array)
    if array.ndim > 2:
        array = np.squeeze(
            array, axis=tuple((i for i in range(1, array.ndim) if array.shape[i] == 1))
        )
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("%s must have nonempty shape [time,dim]; found %s" % (name, array.shape))
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite features in %s" % name)
    return np.asarray(array, dtype=np.float32)


class VideoFeatureStore:

    def __init__(self, path, dataset_name="charades"):
        self.path = Path(path).expanduser().resolve()
        self.dataset_name = canonical_dataset_name(dataset_name)
        if not self.path.exists():
            raise FileNotFoundError("Video feature store is absent: %s" % self.path)
        self.is_directory = self.path.is_dir()
        if self.is_directory:
            self.keys = {p.name[:-4] for p in self.path.glob("*.npy")}
        else:
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("HDF5 features require h5py") from exc
            with h5py.File(self.path, "r") as handle:
                self.keys = set(handle.keys())

    def key_for(self, video_id):
        for key in _video_keys(video_id):
            if key in self.keys:
                return key
        raise KeyError("Missing video %s in %s" % (video_id, self.path))

    def contains(self, video_id):
        return any((key in self.keys for key in _video_keys(video_id)))

    def read(self, video_id):
        key = self.key_for(video_id)
        if self.is_directory:
            array = np.load(self.path / (key + ".npy"), allow_pickle=False)
        else:
            import h5py

            with h5py.File(self.path, "r") as handle:
                node = handle[key]
                if isinstance(node, h5py.Group):
                    candidates = ("c3d_features", "features", "feature", "i3d_features")
                    feature_key = next((item for item in candidates if item in node), None)
                    if feature_key is None:
                        raise KeyError("No feature dataset under HDF5 group %s" % key)
                    node = node[feature_key]
                array = node[:]
        return _feature_array(array, video_id)


class TextFeatureStore:

    def __init__(self, path, max_query_len=32):
        self.path = Path(path).expanduser().resolve()
        self.max_query_len = int(max_query_len)
        if self.max_query_len <= 0:
            raise ValueError("max_query_len must be positive")
        if not self.path.exists():
            raise FileNotFoundError("Text cache absent; run tools/prepare_text.py: %s" % self.path)
        self.is_directory = self.path.is_dir()
        self.manifest = {}
        if self.is_directory and (self.path / "manifest.json").exists():
            self.manifest = json.loads((self.path / "manifest.json").read_text())
            if self.manifest.get("dimension", 768) != 768:
                raise ValueError("PointDiff requires 768-D DistilBERT features")
            if self.manifest.get("max_query_len", self.max_query_len) < self.max_query_len:
                raise ValueError("Text cache was truncated below the requested max_query_len")
        if not self.is_directory:
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("HDF5 text features require h5py") from exc

    def contains(self, key):
        if self.is_directory:
            return (self.path / (key + ".npy")).exists() or (self.path / (key + ".npz")).exists()
        import h5py

        with h5py.File(self.path, "r") as handle:
            return key in handle

    def read(self, key):
        if self.is_directory:
            npy_path = self.path / (key + ".npy")
            if npy_path.exists():
                array = np.load(npy_path, allow_pickle=False)
            else:
                npz_path = self.path / (key + ".npz")
                if not npz_path.exists():
                    raise FileNotFoundError("Missing DistilBERT query cache %s" % npy_path)
                with np.load(npz_path, allow_pickle=False) as data:
                    array = data["features"]
                    if "mask" in data:
                        array = array[np.asarray(data["mask"], dtype=bool)]
        else:
            import h5py

            with h5py.File(self.path, "r") as handle:
                node = handle[key]
                array = node["features"][:] if isinstance(node, h5py.Group) else node[:]
                if isinstance(node, h5py.Group) and "mask" in node:
                    array = array[np.asarray(node["mask"][:], dtype=bool)]
        array = _feature_array(array, key)
        if array.shape[1] != 768:
            raise ValueError("Expected DistilBERT dim768; found %s for %s" % (array.shape, key))
        return torch.from_numpy(array[: self.max_query_len].copy())


def average_pool_video(features, num_clips, normalize=True):
    features = torch.as_tensor(features, dtype=torch.float32)
    if normalize:
        features = F.normalize(features, dim=-1)
    length = features.shape[0]
    boundaries = torch.linspace(0, length, int(num_clips) + 1).round().long()
    pooled = []
    for index in range(int(num_clips)):
        (start, end) = (boundaries[index].item(), boundaries[index + 1].item())
        pooled.append(
            features[start:end].mean(0) if start < end else features[min(start, length - 1)]
        )
    return torch.stack(pooled)


class PointDiffDataset(Dataset):

    def __init__(
        self,
        annotation_file,
        video_features,
        text_features,
        dataset_name="charades",
        training=True,
        num_clips=32,
        max_query_len=32,
        cache_video=True,
        limit: Optional[int] = None,
        normalize_video=True,
        video_cache_capacity=512,
        cache_text=False,
        text_cache_capacity=None,
        include_gpc_video=False,
        gpc_video_cache_capacity=128,
    ):
        self.dataset_name = canonical_dataset_name(dataset_name)
        self.training = bool(training)
        self.num_clips = int(num_clips)
        if self.num_clips <= 0:
            raise ValueError("num_clips must be positive")
        self.records = load_pointdiff_annotations(annotation_file, self.dataset_name, self.training)
        if limit is not None:
            if int(limit) <= 0:
                raise ValueError("limit must be positive")
            self.records = self.records[: int(limit)]
        if not self.records:
            raise ValueError("No queries in %s" % annotation_file)
        self.video_store = VideoFeatureStore(video_features, self.dataset_name)
        self.text_store = TextFeatureStore(text_features, max_query_len)
        missing = sorted(
            {
                item["video_id"]
                for item in self.records
                if not self.video_store.contains(item["video_id"])
            }
        )
        if missing:
            raise FileNotFoundError(
                "%d annotated videos lack features; first: %s" % (len(missing), missing[:5])
            )
        self.video_dim = int(self.video_store.read(self.records[0]["video_id"]).shape[1])
        self.text_dim = 768
        self.normalize_video = bool(normalize_video)
        self.include_gpc_video = bool(include_gpc_video)
        self.gpc_video_cache_capacity = self._validate_capacity(
            gpc_video_cache_capacity, "gpc_video_cache_capacity"
        )
        self.cache_video = bool(cache_video)
        self.cache_text = bool(cache_text)
        self.video_cache_capacity = self._validate_capacity(
            video_cache_capacity, "video_cache_capacity"
        )
        self.text_cache_capacity = self._validate_capacity(
            text_cache_capacity, "text_cache_capacity"
        )
        self._video_cache = OrderedDict()
        self._text_cache = OrderedDict()
        self._gpc_video_cache = OrderedDict()

    @staticmethod
    def _validate_capacity(value, name):
        if value is None:
            return None
        if not isinstance(value, int) or value < 0:
            raise ValueError("%s must be a nonnegative integer or None" % name)
        return value

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        video_id = record["video_id"]
        raw = None
        if video_id in self._video_cache:
            video = self._video_cache.pop(video_id)
            self._video_cache[video_id] = video
        else:
            raw = self.video_store.read(video_id)
            if raw.shape[1] != self.video_dim:
                raise ValueError("Video feature dimension changed for %s" % video_id)
            video = average_pool_video(raw, self.num_clips, self.normalize_video)
            if self.cache_video:
                self._video_cache[video_id] = video
                if (
                    self.video_cache_capacity is not None
                    and len(self._video_cache) > self.video_cache_capacity
                ):
                    self._video_cache.popitem(last=False)
        text_key = record["text_key"]
        if text_key in self._text_cache:
            text = self._text_cache.pop(text_key)
            self._text_cache[text_key] = text
        else:
            text = self.text_store.read(text_key)
            if self.cache_text:
                self._text_cache[text_key] = text
                if (
                    self.text_cache_capacity is not None
                    and len(self._text_cache) > self.text_cache_capacity
                ):
                    self._text_cache.popitem(last=False)
        item = {
            "video": video,
            "text": text,
            "video_mask": torch.ones(video.shape[0], dtype=torch.bool),
            "text_mask": torch.ones(text.shape[0], dtype=torch.bool),
            "duration": record["duration"],
            "video_id": video_id,
            "query": record["query"],
            "query_id": record["query_id"],
        }
        if self.include_gpc_video:
            if video_id in self._gpc_video_cache:
                native = self._gpc_video_cache.pop(video_id)
            else:
                raw = self.video_store.read(video_id) if raw is None else raw
                native = torch.as_tensor(raw, dtype=torch.float32)
                if self.normalize_video:
                    native = F.normalize(native, dim=-1)
            self._gpc_video_cache[video_id] = native
            if self.gpc_video_cache_capacity is not None:
                while len(self._gpc_video_cache) > self.gpc_video_cache_capacity:
                    self._gpc_video_cache.popitem(last=False)
            item["gpc_video"] = native
            item["gpc_video_mask"] = torch.ones(native.shape[0], dtype=torch.bool)
        if self.training:
            item["point"] = torch.tensor(record["point"], dtype=torch.float32)
            item["point_seconds"] = torch.tensor(record["point_seconds"], dtype=torch.float32)
        else:
            item["target"] = torch.tensor(record["target"], dtype=torch.float32)
        return item


def pointdiff_collate(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(items)
    max_video = max((item["video"].shape[0] for item in items))
    max_text = max((item["text"].shape[0] for item in items))
    batch = {
        "video": torch.zeros(batch_size, max_video, items[0]["video"].shape[-1]),
        "text": torch.zeros(batch_size, max_text, 768),
        "video_mask": torch.zeros(batch_size, max_video, dtype=torch.bool),
        "text_mask": torch.zeros(batch_size, max_text, dtype=torch.bool),
        "duration": torch.tensor([item["duration"] for item in items], dtype=torch.float32),
    }
    for index, item in enumerate(items):
        (video_len, text_len) = (item["video"].shape[0], item["text"].shape[0])
        batch["video"][index, :video_len] = item["video"]
        batch["text"][index, :text_len] = item["text"]
        batch["video_mask"][index, :video_len] = item["video_mask"]
        batch["text_mask"][index, :text_len] = item["text_mask"]
    for key in ("video_id", "query", "query_id"):
        batch[key] = [item[key] for item in items]
    batch["sample_ids"] = batch["query_id"]
    is_train = "point" in items[0]
    if any((("point" in item) != is_train for item in items)):
        raise ValueError("A batch cannot mix training and evaluation records")
    if is_train:
        batch["point"] = torch.stack([item["point"] for item in items])
        batch["point_seconds"] = torch.stack([item["point_seconds"] for item in items])
    else:
        batch["target"] = torch.stack([item["target"] for item in items])
    has_native = "gpc_video" in items[0]
    if any((("gpc_video" in item) != has_native for item in items)):
        raise ValueError("A batch cannot mix native-GPC and pooled-only records")
    if has_native:
        max_native = max((item["gpc_video"].shape[0] for item in items))
        batch["gpc_video"] = torch.zeros(batch_size, max_native, items[0]["gpc_video"].shape[-1])
        batch["gpc_video_mask"] = torch.zeros(batch_size, max_native, dtype=torch.bool)
        for index, item in enumerate(items):
            length = item["gpc_video"].shape[0]
            batch["gpc_video"][index, :length] = item["gpc_video"]
            batch["gpc_video_mask"][index, :length] = item["gpc_video_mask"]
    return batch


def pointdiff_data_report(
    annotation_files, video_features, dataset_name="charades", text_features=None, max_query_len=32
):
    store = VideoFeatureStore(video_features, dataset_name)
    text_store = (
        TextFeatureStore(text_features, max_query_len)
        if text_features is not None and Path(text_features).exists()
        else None
    )
    report = {
        "dataset_name": canonical_dataset_name(dataset_name),
        "video_features": str(store.path),
        "feature_videos": len(store.keys),
        "text_features": str(Path(text_features).resolve()) if text_features else None,
        "text_dim": 768,
        "splits": {},
    }
    for split, path in annotation_files.items():
        records = load_pointdiff_annotations(path, dataset_name, training=split == "train")
        video_ids = {item["video_id"] for item in records}
        missing = sorted((video_id for video_id in video_ids if not store.contains(video_id)))
        keys = {item["text_key"] for item in records}
        present = sum((text_store.contains(key) for key in keys)) if text_store else 0
        report["splits"][split] = {
            "annotations": str(Path(path).resolve()),
            "queries": len(records),
            "annotated_videos": len(video_ids),
            "missing_feature_videos": len(missing),
            "missing_video_examples": missing[:10],
            "unique_queries": len(keys),
            "cached_unique_queries": present,
        }
        if split == "train":
            report["splits"][split]["points_outside_duration"] = sum(
                (not 0 <= item["point"] <= 1 for item in records)
            )
            report["splits"][split]["point_labels_preserved"] = True
        available_id = next(
            (item["video_id"] for item in records if store.contains(item["video_id"])), None
        )
        if available_id is not None:
            shape = store.read(available_id).shape
            report["video_dim"] = int(shape[1])
            report["splits"][split]["example_video_shape"] = list(shape)
    return report
