stack_name ?= nlp_template_stack

{%- if zenml_server_url != '' %}
remote-login:
	zenml connect --url "{{zenml_server_url}}"

setup: remote-login
{%- else %}
setup:
{%- endif %}
	pip install -r requirements.txt
	zenml integration install pytorch mlflow s3 gcp aws slack transformers -y

install-stack:
