# {% include 'template/license_header' %}

from typing import Dict, List, Tuple

import numpy as np
from datasets import load_metric


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute the metrics for the model.

    Args:
        eval_pred: The evaluation prediction.

    Returns:
        The metrics for the model.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculate the mertic using the predicted and true value
    accuracy = load_metric("accuracy", trust_remote_code=True).compute(
        predictions=predictions, references=labels
    )
    f1 = load_metric("f1", trust_remote_code=True).compute(
        predictions=predictions, references=labels, average="weighted"
    )
    precision = load_metric("precision", trust_remote_code=True).compute(
        predictions=predictions, references=labels, average="weighted"
    )
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
    }


def find_max_length(dataset: List[str]) -> int:
    """Find the maximum length of the dataset.

    The dataset is a list of strings which are the text samples.
    We need to find the maximum length of the text samples for
    padding.

    Args:
        dataset: The dataset.

    Returns:
        The maximum length of the dataset.
    """
    return len(max(dataset, key=lambda x: len(x.split())).split())
