#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.


import os
import pathlib
import platform
import shutil
import subprocess
import sys
from typing import Optional

import pytest
from copier import Worker
from zenml.client import Client
from zenml.enums import ExecutionStatus

TEMPLATE_DIRECTORY = str(pathlib.Path.joinpath(pathlib.Path(__file__).parent.parent))

def generate_and_run_project(
    tmp_path_factory: pytest.TempPathFactory,
    open_source_license: Optional[str] = "apache",
    product_name: str = "nlp_case_pytest",
    metric_compare_promotion: bool = True,
    target_environment: str = "staging",
    notify_on_failures: bool = True,
    notify_on_successes: bool = False,
    sample_rate: bool = True,
    model: str = "distilbert-base-uncased",
    zenml_server_url: str = "",
    accelerator: str = "cpu",
    deploy_locally: bool = True,
    deploy_to_huggingface: bool = False,
    deploy_to_skypilot: bool = False,
    cloud_of_choice: str = "gcp",
    dataset: str = "airline_reviews",
    zenml_model_name: str = "sentiment_analysis",
):
    """Generate and run the starter project with different options."""

    answers = {
        "project_name": "Pytest Templated Project",
        "version": "0.0.1",
        "open_source_license": str(open_source_license).lower(),
        "product_name": product_name,
        "metric_compare_promotion": metric_compare_promotion,
        "target_environment": target_environment,
        "notify_on_failures": notify_on_failures,
        "notify_on_successes": notify_on_successes,
        "zenml_server_url": zenml_server_url,
        "sample_rate": sample_rate,
        "model": model,
        "accelerator": accelerator,
        "deploy_locally": deploy_locally,
        "deploy_to_huggingface": deploy_to_huggingface,
        "deploy_to_skypilot": deploy_to_skypilot,
        "cloud_of_choice": cloud_of_choice,
        "dataset": dataset,
    }
    if open_source_license:
        answers["email"] = "pytest@zenml.io"
        answers["full_name"] = "Pytest"

    # generate the template in a temp path
    current_dir = os.getcwd()
    dst_path = tmp_path_factory.mktemp("pytest-template")
    print("TEMPLATE_DIR:", TEMPLATE_DIRECTORY)
    print("dst_path:", dst_path)
    print("current_dir:", current_dir)
    os.chdir(str(dst_path))
    with Worker(
        src_path=TEMPLATE_DIRECTORY,
        dst_path=str(dst_path),
        data=answers,
        unsafe=True,
        vcs_ref="HEAD",
    ) as worker:
        worker.run_copy()

    # MLFlow Deployer not supported on Windows
    # MLFlow `service daemon is not running` error on MacOS
    if platform.system().lower() not in ["windows"]:
        # run the project
        call = [sys.executable, "run.py"]

        try:
            subprocess.check_output(
                call,
                cwd=str(dst_path),
                env=os.environ.copy(),
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to run project generated with parameters: {answers}\n"
                f"{e.output.decode()}"
            ) from e

        # check the pipeline run is successful
        for pipeline_suffix in ["_training_pipeline", "_promote_pipeline"]:
            pipeline = Client().get_pipeline(product_name + pipeline_suffix)
            assert pipeline
            runs = pipeline.runs
            assert len(runs) == 1
            assert runs[0].status == ExecutionStatus.COMPLETED

            # clean up
            Client().delete_pipeline(product_name + pipeline_suffix)
        Client().delete_model(zenml_model_name)
        Client().active_stack.model_registry.delete_model(zenml_model_name)

    os.chdir(current_dir)
    shutil.rmtree(dst_path)


@pytest.mark.parametrize("open_source_license", ["mit", None], ids=["oss", "css"])
def test_generate_license(
    clean_zenml_client,
    tmp_path_factory: pytest.TempPathFactory,
    open_source_license: Optional[str],
):
    """Test generating licenses."""

    generate_and_run_project(
        tmp_path_factory=tmp_path_factory,
        open_source_license=open_source_license,
    )


def test_custom_product_name(
    clean_zenml_client,
    tmp_path_factory: pytest.TempPathFactory,
):
    """Test using custom pipeline name."""

    generate_and_run_project(
        tmp_path_factory=tmp_path_factory,
        product_name="custom_product_name",
    )
