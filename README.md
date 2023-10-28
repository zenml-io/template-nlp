# 💫 ZenML End-to-End NLP Training and Deployment Project Template

This project template is designed to help you get started with training and deploying NLP models using the ZenML framework. It provides a comprehensive set of steps and pipelines to cover major use cases of NLP model development, including dataset loading, tokenization, model training, model registration, and deployment.

## 📃 Template Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Name | The name of the person/entity holding the copyright | ZenML GmbH |
| Email | The email of the person/entity holding the copyright | info@zenml.io |
| Project Name | Short name for your project | ZenML NLP project |
| Project Version | The version of your project | 0.0.1 |
| Project License | The license under which your project will be released | Apache Software License 2.0 |
| Technical product name | The technical name to prefix all tech assets (pipelines, models, etc.) | nlp_use_case |
| Target environment | The target environment for deployments/promotions | staging |
| Use metric-based promotion | Whether to compare metric of interest to make model version promotion | True |
| Notifications on failure | Whether to notify about pipeline failures | True |
| Notifications on success | Whether to notify about pipeline successes | False |
| ZenML Server URL | Optional URL of a remote ZenML server for support scripts | - |

## 🚀 Generate a ZenML Project

To generate a project from this template, make sure you have ZenML and its `templates` extras installed:

```bash
pip install zenml[templates]
```

Then, run the following command to generate the project:

```bash
zenml init --template nlp-template
```

You will be prompted to provide values for the template parameters. If you want to use the default values, you can add the `--template-with-defaults` flag to the command.

## 🧰 How this template is implemented

This template provides a set of pipelines and steps to cover the end-to-end process of training and deploying NLP models. Here is an overview of the main components:

### Dataset Loading

The template includes a step for loading the dataset from the HuggingFace Datasets library. You can choose from three available datasets: financial_news, airline_reviews, and imbd_reviews.

### Tokenization

The tokenization step preprocesses the dataset by tokenizing the text data using the tokenizer provided by the HuggingFace Models library. You can choose from three available models: bert-base-uncased, roberta-base, and distilbert-base-cased.

### Model Training

The training pipeline consists of several steps, including model architecture search, hyperparameter tuning, model training, and model evaluation. The best model architecture and hyperparameters are selected based on the performance on the validation set. The trained model is then evaluated on the holdout set to assess its performance.

### Model Registration and Promotion

After training, the best model version is registered in the ZenML Model Registry. The template provides an option to promote the model version based on a specified metric of interest. If metric-based promotion is enabled, the template compares the metric value of the new model version with the metric value of the current production model version and promotes the new version if it performs better.

### Batch Inference

The template includes a batch inference pipeline that loads the inference dataset, preprocesses it using the same tokenizer as during training, and runs predictions using the deployed model version. The predictions are stored as an artifact for future use.

### Deployment Options

The template provides options to deploy the trained model locally or to the HuggingFace Hub. You can choose whether to deploy locally or to the HuggingFace Hub by setting the `deploy_locally` and `deploy_to_huggingface` parameters.

## Next Steps

Once you have generated the project using this template, you can explore the generated code and customize it to fit your specific NLP use case. The README.md file in the generated project provides further instructions on how to set up and run the project.

Happy coding with ZenML and NLP!