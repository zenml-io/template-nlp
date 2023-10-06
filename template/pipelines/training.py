# {% include 'template/license_header' %}


from typing import List, Optional

from config import DEFAULT_PIPELINE_EXTRAS, PIPELINE_SETTINGS, MetaConfig
from steps import (
    data_loader,
    model_trainer,
    notify_on_failure,
    notify_on_success,
    promote_latest,
    promote_get_versions,
    tokenization_step,
    tokenizer_loader,
)
from zenml import pipeline
from zenml.integrations.mlflow.steps.mlflow_deployer import (
    mlflow_model_registry_deployer_step,
)
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step
from zenml.logger import get_logger
from zenml.steps.external_artifact import ExternalArtifact


logger = get_logger(__name__)


@pipeline(
    settings=PIPELINE_SETTINGS,
    on_failure=notify_on_failure,
    extra=DEFAULT_PIPELINE_EXTRAS,
)
def {{product_name}}_training(
    #hf_dataset: HFSentimentAnalysisDataset,
    #hf_tokenizer: HFPretrainedTokenizer,
    #hf_pretrained_model: HFPretrainedModel,
    lower_case: Optional[bool] = True,
    padding: Optional[str] = "max_length",
    max_seq_length: Optional[int] = 128,
    text_column: Optional[str] = "text",
    label_column: Optional[str] = "label",
    train_batch_size: Optional[int] = 16,
    eval_batch_size: Optional[int] = 16,
    num_epochs: Optional[int] = 3,
    learning_rate: Optional[float] = 2e-5,
    weight_decay: Optional[float] = 0.01,
):
    """
    Model training pipeline.

    This is a pipeline that loads the data, processes it and splits
    it into train and test sets, then search for best hyperparameters,
    trains and evaluates a model.

    Args:
        test_size: Size of holdout set for training 0.0..1.0
        drop_na: If `True` NA values will be removed from dataset
        normalize: If `True` dataset will be normalized with MinMaxScaler
        drop_columns: List of columns to drop from dataset
        random_seed: Seed of random generator,
        min_train_accuracy: Threshold to stop execution if train set accuracy is lower
        min_test_accuracy: Threshold to stop execution if test set accuracy is lower
        fail_on_accuracy_quality_gates: If `True` and `min_train_accuracy` or `min_test_accuracy`
            are not met - execution will be interrupted early

    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    ########## Tokenization stage ##########
    dataset = data_loader(
        hf_dataset=MetaConfig.dataset,
    )
    tokenizer = tokenizer_loader(
        hf_tokenizer=MetaConfig.tokenizer, 
        lower_case=lower_case
    )
    tokenized_data = tokenization_step(
        dataset=dataset,
        tokenizer=tokenizer,
        padding=padding,
        max_seq_length=max_seq_length,
        text_column=text_column,
        label_column=label_column
    )


    ########## Training stage ##########
    model = model_trainer(
        tokenized_dataset=tokenized_data,
        hf_pretrained_model=MetaConfig.model,
        tokenizer=tokenizer,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    mlflow_register_model_step(
        model,
        name=MetaConfig.mlflow_model_name,
    )

    ########## Promotion stage ##########
    latest_version, current_version = promote_get_versions(
        after=["mlflow_register_model_step"],
    )
    promote_latest(
        latest_version=latest_version,
        current_version=current_version,
    )
    last_step_name = "promote_latest"

    notify_on_success(after=[last_step_name])
    ### YOUR CODE ENDS HERE ###
