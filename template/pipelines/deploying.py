# {% include 'template/license_header' %}

from typing import Optional, List

from steps import (
    notify_on_failure,
    notify_on_success,
    save_model_to_deploy,
{% if deploy_locally %}
    deploy_locally,
{% endif %}
{% if deploy_to_huggingface %}
    deploy_to_huggingface,
{% endif %}
{% if deploy_to_skypilot %}
    deploy_to_skypilot,
{%- endif %}
)
from zenml import get_pipeline_context, pipeline
from zenml.logger import get_logger
from zenml.client import Client

logger = get_logger(__name__)

# Get experiment tracker
orchestrator = Client().active_stack.orchestrator

# Check if orchestrator flavor is local
if orchestrator.flavor not in ["local", "vm_aws", "vm_gcp", "vm_azure"]:
    raise RuntimeError(
        "Your active stack needs to contain a local orchestrator or a VM "
        "orchestrator to run this pipeline. However, we recommend using "
        "the local orchestrator for this pipeline."
    )

@pipeline(
    on_failure=notify_on_failure,
)
def {{product_name}}_deploy_pipeline(
    labels: Optional[List[str]] = ["Negative", "Positive"],
    title: Optional[str] = None,
    description: Optional[str] = None,
    model_name_or_path: Optional[str] = "model",
    tokenizer_name_or_path: Optional[str] = "tokenizer",
    interpretation: Optional[str] = None,
    example: Optional[str] = None,
    repo_name: Optional[str] = "{{product_name}}",
):
    """
    Model deployment pipeline.

    This pipelines deploys latest model on mlflow registry that matches
    the given stage, to one of the supported deployment targets.

    Args:
        labels: List of labels for the model.
        title: Title for the model.
        description: Description for the model.
        model_name_or_path: Name or path of the model.
        tokenizer_name_or_path: Name or path of the tokenizer.
        interpretation: Interpretation for the model.
        example: Example for the model.
        repo_name: Name of the repository to deploy to HuggingFace Hub.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    ########## Save Model locally ##########
    save_model_to_deploy()

{%- if deploy_locally %}  
    ########## Deploy Locally ##########
    deploy_locally(
        labels=labels,
        title=title,
        description=description,
        interpretation=interpretation,
        example=example,
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        after=["save_model_to_deploy"],
    )

{%- endif %}

{%- if deploy_to_huggingface %}  
    ########## Deploy to HuggingFace ##########
    deploy_to_huggingface(
        repo_name=repo_name,
        after=["save_model_to_deploy"],
    )

{%- endif %}

{%- if deploy_to_skypilot %}  
    ########## Deploy to Skypilot ##########
    deploy_to_skypilot(
        after=["save_model_to_deploy"],
    )

{%- endif %}

{%- if deploy_to_skypilot %} 
    last_step_name = "deploy_to_skypilot"
{%- elif deploy_to_huggingface %}
    last_step_name = "deploy_to_huggingface"
{%- elif deploy_locally %}
    last_step_name = "deploy_locally"
{%- endif %}

    notify_on_success(after=[last_step_name])
    ### YOUR CODE ENDS HERE ###
