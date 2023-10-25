# {% include 'template/license_header' %}

from typing import Optional

import mlflow
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from zenml import step
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Get experiment tracker
model_registry = Client().active_stack.model_registry


@step()
def save_model_locally(
    mlflow_model_name: str,
    stage: str,
):
    """
    

    Args:
        mlfow_model_name: The name of the model in MLFlow.
        stage: The stage of the model in MLFlow.

    Returns:
        The trained model and tokenizer.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Load model from MLFlow registry
    loaded_model = model_registry.load_model_version(
        name=mlflow_model_name,
        version=stage,
    )
    # Save the model and tokenizer locally
    model_path = "./gradio/model"  # replace with the actual path
    tokenizer_path = "./gradio/tokenizer"  # replace with the actual path

    # Save model locally
    model = loaded_model["model"].save_pretrained(model_path)
    tokenizer = loaded_model["tokenizer"].save_pretrained(tokenizer_path)
    ### YOUR CODE ENDS HERE ###
