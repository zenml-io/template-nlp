from typing import Tuple, Optional

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TrainingArguments, Trainer, PreTrainedModel, DataCollatorWithPadding
from transformers import BertForSequenceClassification, GPT2ForSequenceClassification
from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase
import mlflow

from zenml import step
from zenml.enums import StrEnum
from zenml.client import Client

from utils.misc import compute_metrics

experiment_tracker = Client().active_stack.experiment_tracker

class HFPretrainedModel(StrEnum):
    """HuggingFace Sentiment Analysis Model."""
    bert = "bert-base-uncased"
    gpt2 = "gpt2"


@step(experiment_tracker=experiment_tracker.name)
def model_trainer(
    hf_pretrained_model: HFPretrainedModel,
    tokenized_dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    num_labels: Optional[int] = 3,
    train_batch_size: Optional[int] = 16,
    num_epochs: Optional[int] = 3,
    seed: Optional[int] = 42,
    learning_rate: Optional[float] = 2e-5,
    load_best_model_at_end: Optional[bool] = True,
    eval_batch_size: Optional[int] = 16,
    weight_decay: Optional[float] = 0.01,
) -> PreTrainedModel:
    """Configure and train a model on the training dataset.

    This is an example of a model training step that takes in a dataset artifact
    previously loaded and pre-processed by other steps in your pipeline, then
    configures and trains a model on it. The model is then returned as a step
    output artifact.

    Model training steps should have caching disabled if they are not
    deterministic (i.e. if the model training involve some random processes
    like initializing weights or shuffling data that are not controlled by
    setting a fixed random seed). This example step ensures the outcome is
    deterministic by initializing the model with a fixed random seed.

    This step is parameterized using the `ModelTrainerStepParameters` class,
    which allows you to configure the step independently of the step code,
{%- if configurable_model %}
    before running it in a pipeline. In this example, the step can be configured
    to use a different model, change the random seed, or pass different
    hyperparameters to the model constructor. See the documentation for more
    information:
{%- else %}
    before running it in a pipeline. In this example, the step can be configured
    to change the random seed, or pass different hyperparameters to the model
    constructor. See the documentation for more information:
{%- endif %}

        https://docs.zenml.io/user-guide/starter-guide/cache-previous-executions

    Args:
        params: The parameters for the model trainer step.
        train_set: The training data set artifact.

    Returns:
        The trained model artifact.
    """
    mlflow.transformers.autolog()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    if hf_pretrained_model == HFPretrainedModel.bert:
        model = BertForSequenceClassification.from_pretrained(hf_pretrained_model.value, num_labels=num_labels)
        training_args = TrainingArguments(
            output_dir="zenml_artifact",
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            save_strategy="epoch",
            load_best_model_at_end=load_best_model_at_end,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
    elif hf_pretrained_model == HFPretrainedModel.gpt2:
        model = GPT2ForSequenceClassification.from_pretrained(hf_pretrained_model.value, num_labels=3)
        model.resize_token_embeddings(len(tokenizer))
        training_args = TrainingArguments(
            output_dir="zenml_artifact",
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            save_strategy="epoch",
            load_best_model_at_end=load_best_model_at_end,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Initialize the model with the hyperparameters indicated in the step
    # parameters and train it on the training set.

    ### YOUR CODE ENDS HERE ###
    trainer.train()
    trainer.evaluate()
    return model