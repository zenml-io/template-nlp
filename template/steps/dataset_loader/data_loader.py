# {% include 'template/license_header' %}


from typing_extensions import Annotated

from datasets import load_dataset, DatasetDict

from zenml import step
from zenml.logger import get_logger

from config import HFSentimentAnalysisDataset

logger = get_logger(__name__)


@step
def data_loader(
    hf_dataset: HFSentimentAnalysisDataset,
) -> Annotated[DatasetDict, "dataset"]:
    """Data loader step.

    This is an example of a data loader step that is usually the first step
    in your pipeline. It reads data from an external source like a file,
    database or 3rd party library, then formats it and returns it as a step
    output artifact.

    This step is parameterized using the `DataLoaderStepParameters` class, which
    allows you to configure the step independently of the step code, before
    running it in a pipeline. In this example, the step can be configured to
    load different built-in scikit-learn datasets. See the documentation for
    more information:

        https://docs.zenml.io/user-guide/starter-guide/cache-previous-executions

    Data loader steps should have caching disabled if they are not deterministic
    (i.e. if they data they load from the external source can be different when
    they are subsequently called, even if the step code and parameter values
    don't change).

    Args:
        params: Parameters for the data loader step.

    Returns:
        The loaded dataset artifact.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Load the dataset indicated in the step parameters and format it as a
    # pandas DataFrame
    logger.info(f"Loaded dataset {hf_dataset.value}")
    if (
        hf_dataset == HFSentimentAnalysisDataset.financial_news
        or hf_dataset != HFSentimentAnalysisDataset.imbd_reviews
        and hf_dataset == HFSentimentAnalysisDataset.airline_reviews
    ):
        dataset = load_dataset(hf_dataset.value)
    elif hf_dataset == HFSentimentAnalysisDataset.imbd_reviews:
        dataset = load_dataset(hf_dataset.value, split='train')
    logger.info(dataset)
    logger.info("Sample Example :", dataset["train"][1])
    ### YOUR CODE ENDS HERE ###
    
    return dataset