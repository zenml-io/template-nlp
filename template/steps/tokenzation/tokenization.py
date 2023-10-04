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
    max_seq_length: int,
    text_column: str,
    label_column: str,
    tokenizer: PreTrainedTokenizerBase,
    dataset: DatasetDict,
) -> Annotated[DatasetDict, "tokenized_data"]:
    """Data splitter step.

    This is an example of a data splitter step that splits the dataset into
    training and dev subsets to be used for model training and evaluation. It
    takes in a dataset as an step input artifact and returns the training and
    dev subsets as two separate step output artifacts.

    Data splitter steps should have a deterministic behavior, i.e. they should
    use a fixed random seed and always return the same split when called with
    the same input dataset. This is to ensure reproducibility of your pipeline
    runs.

    This step is parameterized using the `DataSplitterStepParameters` class,
    which allows you to configure the step independently of the step code,
    before running it in a pipeline. In this example, the step can be configured
    to use a different random seed, change the split ratio, or control whether
    to shuffle or stratify the split. See the documentation for more
    information:

        https://docs.zenml.io/user-guide/starter-guide/cache-previous-executions

    Args:
        params: Parameters for the data splitter step.
        dataset: The dataset to split.

    Returns:
        The resulting training and dev subsets.
    """
    train_max_length = find_max_length(dataset["train"]["text"])
    val_max_length = find_max_length(dataset["validation"]["text"])
    max_length = train_max_length if train_max_length >= val_max_length else val_max_length
    logger.info(f"max length for the given dataset is:{max_length}")
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    def preprocess_function(examples):
        result = tokenizer(
            examples[text_column],
            #padding="max_length",
            truncation=True,
            max_length=max_length or max_seq_length
        )
        result["label"] = examples[label_column]
        return result
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    #tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    #tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    #tokenized_datasets.set_format("torch")
    return tokenized_datasets