import gradio as gr
from transformers import pipeline

# Load the model
model = pipeline("summarization")


def predict(text):
    return model(text)[0]["summary_text"]


# Create a Gradio interface to summarize text with a text box input
with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Input Text")
    output_text = gr.Textbox(label="Summary")
    summarize_button = gr.Button("Summarize")

    summarize_button.click(
        fn=predict, inputs=input_text, outputs=output_text
    )  # pylint: disable=no-member

demo.launch()
