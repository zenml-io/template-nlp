# {% include 'template/license_header' %}

from typing import Optional

from steps import (
    notify_on_failure,
    notify_on_success,
{%- if metric_compare_promotion %}
    promote_get_metric,
    promote_metric_compare_promoter,
{%- else %}
    promote_latest,
{%- endif %}
    promote_get_versions,
)
from zenml import get_pipeline_context
from zenml.logger import get_logger

logger = get_logger(__name__)

# Get experiment tracker
orchestrator = Client().active_stack.orchestrator

# Check if orchestrator flavor is either default or skypilot
if orchestrator.flavor not in ["default"]:
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
    deploy_local(
        model="{{model}}",
        labels=labels,
        title=title,
        description=description,
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
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    last_step_name = "deploy_to_skypilot"
{%- endif %}

    notify_on_success(after=[last_step_name])
    ### YOUR CODE ENDS HERE ###
