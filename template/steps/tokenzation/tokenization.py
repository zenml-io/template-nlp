# {% include 'template/license_header' %}

from typing_extensions import Annotated
from transformers import PreTrainedTokenizerBase
from datasets import DatasetDict
from zenml import step
from zenml.logger import get_logger
from utils.misc import find_max_length

logger = get_logger(__name__)

@step
def tokenization_step(
    padding: str,
    max_seq_length: int = 512,
    text_column: str = "text",
    label_column: str = "label",
    tokenizer: PreTrainedTokenizerBase,
    dataset: DatasetDict,
) -> Annotated[DatasetDict, "tokenized_data"]:
    """
    Tokenization step.

    This step tokenizes the input dataset using a pre-trained tokenizer. It
    takes in a dataset as an step input artifact and returns the tokenized
    dataset as an output artifact.

    The tokenization process includes padding, truncation, and addition of
    labels. The maximum sequence length for tokenization is determined based
    on the dataset.

    This step is parameterized using the `DataSplitterStepParameters` class,
    which allows you to configure the step independently of the step code,
    before running it in a pipeline. In this example, the step can be configured
    to use a different random seed, change the split ratio, or control whether
    to shuffle or stratify the split. See the documentation for more
    information:

        https://docs.zenml.io/user-guide/starter-guide/cache-previous-executions

    Args:
        padding: Padding method for tokenization.
        max_seq_length: Maximum sequence length for tokenization.
        text_column: Column name for text data in the dataset.
        label_column: Column name for label data in the dataset.
        tokenizer: Pre-trained tokenizer for tokenization.
        dataset: The dataset to tokenize.

    Returns:
        The tokenized dataset.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    train_max_length = find_max_length(dataset["train"][text_column])

    # Depending on the dataset, find the maximum length of text in the validation or test dataset
{%- if dataset == 'imdb' %}
    val_or_test_max_length = find_max_length(dataset["test"][text_column])
{%- else %}
    val_or_test_max_length = find_max_length(dataset["validation"][text_column])
    max_length = train_max_length if train_max_length >= val_or_test_max_length else val_or_test_max_length
{%- endif %}
    logger.info(f"max length for the given dataset is:{max_length}")

    # Determine the maximum length for tokenization
    max_length = train_max_length if train_max_length >= val_or_test_max_length else val_or_test_max_length
    logger.info(f"max length for the given dataset is:{max_length}")

    def preprocess_function(examples):
        # Tokenize the examples with padding, truncation, and a specified maximum length
        result = tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length or max_seq_length,
        )
        # Add labels to the tokenized examples
        result["label"] = examples[label_column]
        return result

    # Apply the preprocessing function to the dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True,)
    logger.info(tokenized_datasets)

    # Remove the original text column and rename the label column
    tokenized_datasets = tokenized_datasets.remove_columns([text_column])
    tokenized_datasets = tokenized_datasets.rename_column(label_column, "labels")

    # Set the format of the tokenized dataset
    tokenized_datasets.set_format("torch")
    ### YOUR CODE ENDS HERE ###
    return tokenized_datasets
