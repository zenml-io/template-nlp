# {% include 'template/license_header' %}

from typing import Optional

import mlflow
from zenml import step
from zenml.client import Client
from zenml.model_registries.base_model_registry import ModelVersionStage
from zenml.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Get experiment tracker
model_registry = Client().active_stack.model_registry


@step()
def save_model_to_deploy(
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
    logger.info(f" Loading latest version of model {mlflow_model_name} for stage {stage}...")
    # Load model from MLFlow registry
    model_version = model_registry.get_latest_model_version(
        name=mlflow_model_name,
        stage=ModelVersionStage(stage),
    ).version
    # Load model from MLFlow registry
    model_version = model_registry.get_model_version(
        name=mlflow_model_name,
        version=model_version,
    )
    transformer_model = mlflow.transformers.load_model(model_version.model_source_uri)
    # Save the model and tokenizer locally
    model_path = "./gradio/model"  # replace with the actual path
    tokenizer_path = "./gradio/tokenizer"  # replace with the actual path

    # Save model locally
    transformer_model.model.save_pretrained(model_path)
    transformer_model.tokenizer.save_pretrained(tokenizer_path)
    logger.info(f" Model and tokenizer saved to {model_path} and {tokenizer_path} respectively.")
    ### YOUR CODE ENDS HERE ###
