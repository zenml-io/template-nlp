# {% include 'template/license_header' %}

from typing import Optional

from steps import (
    notify_on_failure,
    notify_on_success,
{%- if metric_compare_promotion %}
    promote_get_metrics,
    promote_metric_compare_promoter,
{%- else %}
    promote_current,
{%- endif %}
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
    ########## Promotion stage ##########
{%- if metric_compare_promotion %}
    latest_metrics, current_metrics = promote_get_metrics()

    promote_metric_compare_promoter(
        latest_metrics=latest_metrics,
        current_metrics=current_metrics,
    )
    last_step_name = "promote_metric_compare_promoter"
{%- else %}
    promote_current()
    last_step_name = "promote_current"
{%- endif %}

    notify_on_success(after=[last_step_name])
    ### YOUR CODE ENDS HERE ###
