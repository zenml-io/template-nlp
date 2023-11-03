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
from zenml import pipeline, get_pipeline_context
from zenml.logger import get_logger


logger = get_logger(__name__)

@pipeline(
    on_failure=notify_on_failure,
)
def {{product_name}}_promote_pipeline():
    """
    Model promotion pipeline.

    This is a pipeline that promotes the best model to the chosen
    stage, e.g. Production or Staging. Based on a metric comparison
    between the latest and the currently promoted model version,
    or just the latest model version.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    pipeline_extra = get_pipeline_context().extra

    ########## Promotion stage ##########
    latest_version, current_version = promote_get_versions()
{%- if metric_compare_promotion %}
    latest_metric = promote_get_metric(
        name=pipeline_extra["mlflow_model_name"],
        metric="eval_loss",
        version=latest_version,
    )
    current_metric = promote_get_metric(
        name=pipeline_extra["mlflow_model_name"],
        metric="eval_loss",
        version=current_version,
    )

    promote_metric_compare_promoter(
        latest_metric=latest_metric,
        current_metric=current_metric,
        latest_version=latest_version,
        current_version=current_version,
    )
    last_step_name = "promote_metric_compare_promoter"
{%- else %}
    promote_latest(
         latest_version=latest_version,
        current_version=current_version,
    )
    last_step_name = "promote_latest"
{%- endif %}

    notify_on_success(after=[last_step_name])
    ### YOUR CODE ENDS HERE ###
