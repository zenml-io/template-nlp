# {% include 'template/license_header' %}


from typing import Optional

from steps import (
    notify_on_failure,
    notify_on_success,
    data_loader,
    tokenizer_loader,
    tokenization_step,
    model_trainer,
    register_model,
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

    This is a pipeline that loads the dataset and tokenizer,
    tokenizes the dataset, trains a model and registers the model
    to the model registry.

    Args:
        lower_case: Whether to convert all text to lower case.
        padding: Padding strategy.
        max_seq_length: Maximum sequence length.
        text_column: Name of the text column.
        label_column: Name of the label column.
        train_batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        num_epochs: Number of epochs.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    ########## Load Dataset stage ##########
    dataset = data_loader()

    ########## Tokenization stage ##########
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
    register_model(
        model=model,
        tokenizer=tokenizer,
        mlflow_model_name="sentiment_analysis",
    )

    notify_on_success(after=["register_model"])
    ### YOUR CODE ENDS HERE ###
