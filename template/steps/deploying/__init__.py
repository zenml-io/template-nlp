# {% include 'template/license_header' %}


from .save_model import save_model_to_deploy
{% if deployment_platform == "local" %}
from .local_deployment import deploy_locally
{% endif %}
{% if deployment_platform == "huggingface" %}
from .huggingface_deployment import deploy_to_huggingface
{% endif %}
{% if deployment_platform == "skypilot" %}
from .skypilot_deployment import deploy_to_skypilot
{%- endif %}