# {% include 'template/license_header' %}


from typing import Optional

from steps import (
    notify_on_failure,
    notify_on_success,
    data_loader,
    tokenizer_loader,
    tokenization_step,
    model_trainer,
    model_log_register,
{%- if metric_compare_promotion %}
    promote_get_metric,
    promote_metric_compare_promoter,
{%- else %}
    promote_latest,
{%- endif %}
)
from zenml import pipeline, get_pipeline_context
from zenml.logger import get_logger


logger = get_logger(__name__)


@pipeline(on_failure=notify_on_failure)
def {{product_name}}_training_pipeline(
    lower_case: Optional[bool] = True,
    padding: Optional[str] = "max_length",
    max_seq_length: Optional[int] = 128,
    text_column: Optional[str] = "text",
    label_column: Optional[str] = "label",
    train_batch_size: Optional[int] = 8,
    eval_batch_size: Optional[int] = 8,
    num_epochs: Optional[int] = 5,
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
    pipeline_extra = get_pipeline_context().extra
    ########## Tokenization stage ##########
    dataset = data_loader(
        shuffle=True,
    )
    tokenizer = tokenizer_loader(
        lower_case=lower_case
    )
    tokenized_data = tokenization_step(
        dataset=dataset,
        tokenizer=tokenizer,
        padding=padding,
        max_seq_length=max_seq_length,
        text_column=text_column,
        label_column=label_column,
    )

    ########## Training stage ##########
    model, tokenizer = model_trainer(
        tokenized_dataset=tokenized_data,
        tokenizer=tokenizer,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    ########## Log and Register stage ##########
    model_log_register(
        model=model,
        tokenizer=tokenizer,
        name="{{product_name}}_model",
    )

    notify_on_success(after=[model_log_register])
    ### YOUR CODE ENDS HERE ###