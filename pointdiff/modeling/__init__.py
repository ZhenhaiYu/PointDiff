from .model import PointDiff


def model_config(config):
    values = dict(config)
    if int(values.pop("implementation_revision", 4)) != 4:
        raise ValueError("This package supports native-grid GPC checkpoints only")
    required = {
        "condition_prior": "none",
        "matching_method": "momentdiff_hungarian",
        "noise_loss_mode": "auxiliary",
        "gaussian_coordinates": "frame_index",
        "gpc_boundary_mode": "point_component",
        "sampling_method": "momentdiff_ddim",
        "loss_span_bounds": "unclipped",
    }
    for name, expected in required.items():
        if values.get(name, expected) != expected:
            raise ValueError(f"Unsupported model setting: {name}={values[name]!r}")
    return values


def build_model(config):
    return PointDiff(model_config(config["model"]))


__all__ = ["PointDiff", "build_model", "model_config"]
