# {% include 'template/license_header' %}

from typing import Optional

from steps import (
    notify_on_failure,
    notify_on_success,
    save_model_to_deploy,
{% if deployment_platform == "local" %}
    deploy_locally,
{% endif %}
{% if deployment_platform == "huggingface" %}
    deploy_to_huggingface,
{% endif %}
{% if deployment_platform == "skypilot" %}
    deploy_to_skypilot,
{%- endif %}
)
from zenml import get_pipeline_context, pipeline
from zenml.logger import get_logger
from zenml.client import Client

logger = get_logger(__name__)

# Get experiment tracker
orchestrator = Client().active_stack.orchestrator

# Check if orchestrator flavor is either default or skypilot
if orchestrator.flavor not in ["local", "vm_aws", "vm_gcp"]:
    raise RuntimeError(
        "Your active stack needs to contain a default or skypilot orchestrator for "
        "the deployment pipeline to work."
    )

@pipeline(
    on_failure=notify_on_failure,
)
def {{product_name}}_{{deployment_platform}}_deploy_pipeline(
    labels: Optional[dict] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    model_name_or_path: Optional[str] = None,
    tokenizer_name_or_path: Optional[str] = None,
    interpretation: Optional[str] = None,
    example: Optional[str] = None,
):
    """
    Model deployment pipeline.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    pipeline_extra = get_pipeline_context().extra
    ########## Promotion stage ##########
    save_model_to_deploy(
        mlflow_model_name=pipeline_extra["mlflow_model_name"],
        stage=pipeline_extra["target_env"],
    )
{%- if deployment_platform == "local" %}  
    deploy_locally(
        model="{{model}}",
        labels=labels,
        title=title,
        description=description,
        interpretation=interpretation,
        example=example,
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    last_step_name = "deploy_local"
{%- endif %}
{%- if deployment_platform == "huggingface" %}  
    deploy_to_huggingface(
        repo_name="{{project_name}}",
        labels=labels,
        title=title,
        description=description,
    )
    last_step_name = "deploy_to_huggingface"
{%- endif %}
{%- if deployment_platform == "skypilot" %}  
    deploy_to_skypilot(
        model="{{model}}",
        labels=labels,
        title=title,
        description=description,
        interpretation=interpretation,
        example=example,
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    last_step_name = "deploy_to_skypilot"
{%- endif %}

    notify_on_success(after=[last_step_name])
    ### YOUR CODE ENDS HERE ###
