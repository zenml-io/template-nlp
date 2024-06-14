# {% include 'template/license_header' %}

from typing_extensions import Annotated
from datasets import load_dataset, DatasetDict
from zenml import step
from zenml.logger import get_logger
{%- if sample_rate %}
import numpy as np
{%- endif %}

logger = get_logger(__name__)

@step
def data_loader(
) -> Annotated[DatasetDict, "dataset"]:
    """
    Data loader step.

    This step reads data from a Huggingface dataset or a CSV files and returns
    a Huggingface dataset.

    Data loader steps should have caching disabled if they are not deterministic
    (i.e. if they data they load from the external source can be different when
    they are subsequently called, even if the step code and parameter values
    don't change).

    Returns:
        The loaded dataset artifact.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    logger.info("Loading dataset {{dataset}}... ")

    # Load dataset based on the dataset value
    {%- if dataset == 'financial_news' %}
    dataset = load_dataset(
        "zeroshot/twitter-financial-news-sentiment",
        trust_remote_code=True,
    )
    {%- endif %}
    {%- if dataset == 'imdb_reviews' %}
    dataset = load_dataset(
        "imdb",
        trust_remote_code=True,
    )["train"]
    dataset = dataset.train_test_split(test_size=0.25, shuffle=True)
    {%- endif %}
    {%- if dataset == 'airline_reviews' %}
    dataset = load_dataset(
        "Shayanvsf/US_Airline_Sentiment",
        trust_remote_code=True,
    )
    dataset = dataset.rename_column("airline_sentiment", "label")
    dataset = dataset.remove_columns(["airline_sentiment_confidence","negativereason_confidence"])
    {%- endif %}

    {%- if sample_rate %}
    # Sample 20% of the data randomly for the demo
    def sample_dataset(dataset, sample_rate=0.2):
        sampled_dataset = DatasetDict()
        for split in dataset.keys():
            split_size = len(dataset[split])
            indices = np.random.choice(split_size, int(split_size * sample_rate), replace=False)
            sampled_dataset[split] = dataset[split].select(indices)
        return sampled_dataset

    dataset = sample_dataset(dataset)
    {%- endif %}

    # Log the dataset and sample examples
    logger.info(dataset)
    logger.info(f"Sample Example 1 : {dataset['train'][0]['text']} with label {dataset['train'][0]['label']}")
    logger.info(f"Sample Example 1 : {dataset['train'][1]['text']} with label {dataset['train'][1]['label']}")
    logger.info("Dataset Loaded Successfully")
    ### YOUR CODE ENDS HERE ###

    return dataset