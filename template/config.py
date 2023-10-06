# {% include 'template/license_header' %}


from artifacts.model_metadata import ModelMetadata
from pydantic import BaseConfig

from zenml.config import DockerSettings
from utils.misc import (
    HFSentimentAnalysisDataset,
    HFPretrainedModel,
    HFPretrainedTokenizer,
)
from zenml.integrations.constants import (
{%- if cloud_of_choice == 'aws' %}
    SKYPILOT_AWS,
    AWS,
    S3,
{%- endif %}
{%- if cloud_of_choice == 'gcp' %}
    SKYPILOT_GCP,
    GCP,
{%- endif %}
    HUGGINGFACE,
    PYTORCH,
    MLFLOW,
    SLACK,
    
)
from zenml.model_registries.base_model_registry import ModelVersionStage

PIPELINE_SETTINGS = dict(
    docker=DockerSettings(
        required_integrations=[
            {%- if cloud_of_choice == 'aws' %}
                SKYPILOT_AWS,
                AWS,
                S3,
            {%- endif %}
            {%- if cloud_of_choice == 'gcp' %}
                SKYPILOT_GCP,
                GCP,
            {%- endif %}
                HUGGINGFACE,
                PYTORCH,
                MLFLOW,
                SLACK,
        ],
        requirements=[
            "accelerate",
        ],
    ) 
)

DEFAULT_PIPELINE_EXTRAS = dict(
    notify_on_success={{notify_on_successes}}, 
    notify_on_failure={{notify_on_failures}}
)

class MetaConfig(BaseConfig):
{%- if dataset == 'imbd_reviews' %}
    dataset = HFSentimentAnalysisDataset.imbd_reviews
{%- endif %}
{%- if dataset == 'airline_reviews' %}
    dataset = HFSentimentAnalysisDataset.airline_reviews
{%- else %}
    dataset = HFSentimentAnalysisDataset.financial_news
{%- endif %}
{%- if model == 'gpt2' %}
    tokenizer = HFPretrainedTokenizer.gpt2
    model = HFPretrainedModel.gpt2
{%- else %}
    tokenizer = HFPretrainedTokenizer.bert
    model = HFPretrainedModel.bert
{%- endif %}
    pipeline_name_training = "{{product_name}}_training"
    mlflow_model_name = "{{product_name}}_model"
{%- if target_environment == 'production' %}
    target_env = ModelVersionStage.PRODUCTION
{%- else %}
    target_env = ModelVersionStage.STAGING
{%- endif %}

    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###