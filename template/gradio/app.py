# {% include 'template/license_header' %}

import click
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import gradio as gr


@click.command()
@click.option("--tokenizer_name_or_path", default="roberta-base", help="Name or the path of the tokenizer.")
@click.option("--model_name_or_path", default="./gradio/model", help="Name or the path of the model.")
@click.option(
    "--labels", default="Negative,Positive", help="Comma-separated list of labels."
)
@click.option(
    "--title", default="ZenML NLP Use-Case", help="Title of the Gradio interface."
)
@click.option(
    "--description",
    default="Sentiment Analyzer",
    help="Description of the Gradio interface.",
)
@click.option(
    "--interpretation",
    default="default",
    help="Interpretation mode for the Gradio interface.",
)
@click.option(
    "--examples",
    default="This is an awesome journey, I love it!",
    help="Comma-separated list of examples to show in the Gradio interface.",
)
def sentiment_analysis(
    tokenizer_name_or_path, model_name_or_path, labels, title, description, interpretation, examples
):
    labels = labels.split(",")
    examples = [examples.split(",")]

    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def analyze_text(text):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors="pt")
        output = model(**encoded_input)
        scores_ = output[0][0].detach().numpy()
        scores_ = softmax(scores_)

        scores = {l: float(s) for (l, s) in zip(labels, scores_)}
        return scores

    demo = gr.Interface(
        fn=analyze_text,
        inputs=[gr.TextArea("Write your text or tweet here", label="Analyze Text")],
        outputs=["label"],
        title=title,
        description=description,
        interpretation=interpretation,
        examples=examples,
    )

    demo.launch(share=True, debug=True)


if __name__ == "__main__":
    sentiment_analysis()
