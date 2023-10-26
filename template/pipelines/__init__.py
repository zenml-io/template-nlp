# {% include 'template/license_header' %}


from .training import {{product_name}}_training_pipeline
from .promoting import {{product_name}}_promote_pipeline
from .deploying import {{product_name}}_{{deployment_platform}}_deploy_pipeline
