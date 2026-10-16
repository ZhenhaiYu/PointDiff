import math
from .postprocessing import postprocessing_config

RECALL_METRICS = ("R@1_IoU=0.5", "R@1_IoU=0.7", "R@5_IoU=0.5", "R@5_IoU=0.7")


def evaluation_settings(config):
    steps = config["model"]["sampling_steps"]
    if (
        not isinstance(steps, int)
        or isinstance(steps, bool)
        or not 1 <= steps <= config["model"].get("timesteps", 1000)
    ):
        raise ValueError("model.sampling_steps must be an integer in [1, timesteps]")
    return {
        "sampling_steps": steps,
        "postprocessing": postprocessing_config(config.get("evaluation", {}).get("postprocessing")),
    }


def format_metrics(metrics):
    return "R@1(0.5) %.2f | R@1(0.7) %.2f | R@5(0.5) %.2f | R@5(0.7) %.2f | mIoU %.2f" % tuple(
        metrics[key] for key in (*RECALL_METRICS, "mIoU")
    )


def selection_record(epoch, settings, metric, metrics):
    value = float(metrics[metric])
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite validation selection metric: " + metric)
    return {
        "epoch": epoch,
        "metric": metric,
        "value": value,
        "metrics": metrics,
        **settings,
    }
