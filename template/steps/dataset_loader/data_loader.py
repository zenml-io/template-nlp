# {% include 'template/license_header' %}

from typing_extensions import Annotated
from datasets import load_dataset, DatasetDict
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def data_loader(
    shuffle: bool = True,
) -> Annotated[DatasetDict, "dataset"]:
    """
    Data loader step.

    This step reads data from an external source like a file,
    database or 3rd party library, then formats it and returns it as a step
    output artifact.

    This step is parameterized using the `HFSentimentAnalysisDataset` class, which
    allows you to configure the step independently of the step code, before
    running it in a pipeline. In this example, the step can be configured to
    load different built-in scikit-learn datasets.

    Data loader steps should have caching disabled if they are not deterministic
    (i.e. if they data they load from the external source can be different when
    they are subsequently called, even if the step code and parameter values
    don't change).

    Returns:
        The loaded dataset artifact.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    logger.info(f"Loaded dataset {{dataset}}")

    # Load dataset based on the dataset value
    {%- if dataset == 'financial_news' %}
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    {%- endif %}
    {%- if dataset == 'imbd_reviews' %}
    dataset = load_dataset("imdb")["train"]
    dataset = dataset.train_test_split(test_size=0.25, shuffle=shuffle)
    {%- endif %}
    {%- if dataset == 'airline_reviews' %}
    dataset = load_dataset("Shayanvsf/US_Airline_Sentiment")
    dataset = dataset.rename_column("airline_sentiment", "label")
    dataset = dataset.remove_columns(["airline_sentiment_confidence","negativereason_confidence"])
    {%- endif %}

    # Log the dataset and sample examples
    logger.info(dataset)
    logger.info("Sample Example 1 :", dataset["train"][0])
    logger.info("Sample Example 2 :", dataset["train"][1])
    ### YOUR CODE ENDS HERE ###

    return dataset