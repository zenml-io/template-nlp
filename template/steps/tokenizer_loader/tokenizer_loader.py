# {% include 'template/license_header' %}


from typing_extensions import Annotated
from transformers import BertTokenizer, GPT2Tokenizer, PreTrainedTokenizerBase

from zenml.enums import StrEnum
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

class HFPretrainedTokenizer(StrEnum):
    """HuggingFace Sentiment Analysis datasets."""
    bert = "bert-base-uncased"
    gpt2 = "gpt2"

@step
def tokenizer_loader(
    hf_tokenizer: HFPretrainedTokenizer,
    lower_case: bool,
) -> Annotated[PreTrainedTokenizerBase, "tokenzer"]:
    """Tokenizer loader step.

    This is an example of a data processor step that prepares the data so that
    it is suitable for model training. It takes in a dataset as an input step
    artifact and performs any necessary preprocessing steps like cleaning,
    feature engineering, feature selection, etc. It then returns the processed
    dataset as a step output artifact.

    This step is parameterized using the `DataProcessorStepParameters` class,
    which allows you to configure the step independently of the step code,
    before running it in a pipeline. In this example, the step can be configured
    to perform or skip different preprocessing steps (e.g. dropping rows with
    missing values, dropping columns, normalizing the data, etc.). See the
    documentation for more information:

        https://docs.zenml.io/user-guide/starter-guide/cache-previous-executions

    Args:
        params: Parameters for the data processor step.

    Returns:
        The processed dataset artifact.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    if hf_tokenizer == HFPretrainedTokenizer.bert:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=lower_case)
    elif hf_tokenizer == HFPretrainedTokenizer.gpt2:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    ### YOUR CODE ENDS HERE ###

    return tokenizer