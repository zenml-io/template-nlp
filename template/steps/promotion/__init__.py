# {% include 'template/license_header' %}


{%- if metric_compare_promotion %}
from .promote_get_metrics import promote_get_metrics
from .promote_metric_compare_promoter import promote_metric_compare_promoter
{%- else %}
from .promote_current import promote_current
{%- endif %}
