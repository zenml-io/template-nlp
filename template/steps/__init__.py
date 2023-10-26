# {% include 'template/license_header' %}


from .alerts import notify_on_failure, notify_on_success
from .dataset_loader import (
    data_loader,
)
from .promotion import (
{%- if metric_compare_promotion %}
    promote_get_metric,
    promote_metric_compare_promoter,
{%- else %}
    promote_latest,
{%- endif %}
    promote_get_versions,
)
from .registrer import register_model
from .tokenizer_loader import (
    tokenizer_loader,
)
from .tokenzation import (
    tokenization_step,
)
from .training import model_trainer

from .deploying import (
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
