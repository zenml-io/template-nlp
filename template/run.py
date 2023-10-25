# {% include 'template/license_header' %}


from zenml.steps.external_artifact import ExternalArtifact
from zenml.logger import get_logger
from pipelines import {{product_name}}_training
import click
from datetime import datetime as dt

logger = get_logger(__name__)


@click.command(
    help="""
{{ project_name }} CLI v{{ version }}.

Run the {{ project_name }} model training pipeline with various
options.

Examples:


  \b
  # Run the pipeline with default options
  python run.py
               
  \b
  # Run the pipeline without cache
  python run.py --no-cache

  \b
  # Run the pipeline without NA drop and normalization, 
  # but dropping columns [A,B,C] and keeping 10% of dataset 
  # as test set.
  python run.py --num-epochs 3 --train-batch-size 8 --eval-batch-size 8

  \b
  # Run the pipeline with Quality Gate for accuracy set at 90% for train set 
  # and 85% for test set. If any of accuracies will be lower - pipeline will fail.
  python run.py --min-train-accuracy 0.9 --min-test-accuracy 0.85 --fail-on-accuracy-quality-gates


"""
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--num-epochs",
    default=5,
    type=click.INT,
    help="Number of epochs to train the model for.",
)
@click.option(
    "--seed",
    default=42,
    type=click.INT,
    help="Seed for the random number generator.",
)
@click.option(
    "--train-batch-size",
    default=16,
    type=click.INT,
    help="Batch size for training the model.",
)
@click.option(
    "--eval-batch-size",
    default=16,
    type=click.INT,
    help="Batch size for evaluating the model.",
)
@click.option(
    "--learning-rate",
    default=2e-5,
    type=click.FLOAT,
    help="Learning rate for training the model.",
)
@click.option(
    "--weight-decay",
    default=0.01,
    type=click.FLOAT,
    help="Weight decay for training the model.",
)
def main(
    no_cache: bool = False,
    seed: int = 42,
    num_epochs: int = 5,
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments)
      * launching the pipeline

    Args:
        no_cache: If `True` cache will be disabled.
    """

    # Run a pipeline with the required parameters. This executes
    # all steps in the pipeline in the correct order using the orchestrator
    # stack component that is configured in your active ZenML stack.
    pipeline_args = {}
    if no_cache:
        pipeline_args["enable_cache"] = False

    # Execute Training Pipeline
    run_args_train = {
        "num_epochs": num_epochs,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }

    pipeline_args[
        "run_name"
    ] = f"{{product_name}}_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    {{product_name}}_training.with_options(**pipeline_args)(**run_args_train)
    logger.info("Training pipeline finished successfully!")


if __name__ == "__main__":
    main()
