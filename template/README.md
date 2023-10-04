# {{project_name}}

This is a comprehensive NLP project built with the
ZenML framework and its integration. The project trains one or more
scikit-learn classification models to make predictions on the tabular
classification datasets provided by the scikit-learn library. The project was
generated from the [NLP ZenML project template](https://github.com/zenml-io/nlp-template)
with the following properties:
- Project name: {{project_name}}
- Technical Name: {{product_name}}
- Version: `{{version}}`
{%- if open_source_license %}
- Licensed with {{open_source_license}} to {{full_name}}<{{email}}>
{%- endif %}
- Deployment environment: `{{target_environment}}`
{%- if zenml_server_url!='' %}
- Remote ZenML Server URL: `{{zenml_server_url}}`
{%- endif %}

Settings of your project are:
- Dataset: `{{dataset}}`
- Model: `{{model}}`
- Every trained model will be promoted to `{{target_environment}}`
{%- if notify_on_failures and notify_on_successes %}
- Notifications about failures and successes enabled
{%- elif notify_on_failures %}
- Notifications about failures enabled
{%- elif notify_on_successes %}
- Notifications about success enabled
{%- else %}
- All notifications disabled
{%- endif %}

## 👋 Introduction

Welcome to your newly generated "{{project_name}}" project! This is
a great way to get hands-on with ZenML using production-like template. 
The project contains a collection of standard and custom ZenML steps, 
pipelines and other artifacts and useful resources that can serve as a 
solid starting point for your smooth journey with ZenML.

What to do first? You can start by giving the the project a quick run. The
project is ready to be used and can run as-is without any further code
changes! You can try it right away by installing ZenML, the needed
ZenML integration and then calling the CLI included in the project. We also
recommend that you start the ZenML UI locally to get a better sense of what
is going on under the hood:

```bash
# Set up a Python virtual environment, if you haven't already
python3 -m venv .venv
source .venv/bin/activate
# Install requirements & integrations
make setup
# Optionally, provision default local stack
make install-stack
# Start the ZenML UI locally (recommended, but optional);
# the default username is "admin" with an empty password
zenml up
# Run the pipeline included in the project
python run.py
```

When the pipelines are done running, you can check out the results in the ZenML
UI by following the link printed in the terminal (or you can go straight to
the [ZenML UI pipelines run page](http://127.0.0.1:8237/workspaces/default/all-runs?page=1).

Next, you should:

* look at the CLI help to see what you can do with the project:
```bash
python run.py --help
```
