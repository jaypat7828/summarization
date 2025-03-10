import gradio as gr
from transformers import pipeline

# Load the model
model = pipeline("summarization", model="facebook/bart-large-cnn")

# Load example text
with open("example.txt", "r", encoding="utf-8") as file:
    example_text = file.read()


def predict(text, max_length=200):
    summary = model(text, max_length=max_length)[0]["summary_text"]
    return summary


def reset():
    return "", "", "", gr.update(interactive=True)


def get_token_length(text):
    token_length = len(text.split())
    color = "red" if token_length > 750 else "green"
    button_state = gr.update(interactive=token_length <= 750)
    return (
        f'<span style="color:{color}">Input Token Length: {token_length}</span>',
        button_state,
    )


def load_example():
    return example_text


# Create a Gradio interface to summarize text with a text box input
with gr.Blocks() as demo:
    gr.Markdown("# 📝 Text Summarization App")
    gr.Markdown("### ✨ Enter text to get a summary")

    load_example_button = gr.Button("📄 Load Example", size="sm")
    token_length_label = gr.HTML(value="Input Token Length: 0")
    input_text = gr.Textbox(label="Input Text")
    with gr.Row():
        summarize_button = gr.Button("🔍 Summarize", interactive=True, size="medium")
        reset_button = gr.Button("🔄 Reset", size="medium")
    output_text = gr.Textbox(label="Summary")
    with gr.Row():
        feedback_button = gr.Button("👍 Provide Feedback", size="medium")

    input_text.change(  # pylint: disable=E1101
        fn=get_token_length,
        inputs=input_text,
        outputs=[token_length_label, summarize_button],
    )

    summarize_button.click(  # pylint: disable=E1101
        fn=predict,
        inputs=input_text,
        outputs=output_text,
    )
    reset_button.click(  # pylint: disable=E1101
        fn=reset,
        inputs=None,
        outputs=[input_text, output_text, token_length_label, summarize_button],
    )

    load_example_button.click(  # pylint: disable=E1101
        fn=load_example,
        inputs=None,
        outputs=input_text,
    )

    gr.Markdown(
        "[Visit Bart Model](https://huggingface.co/facebook/bart-large-cnn) | [Visit Project](https://github.com/jaypat7828/summarization)"
    )

demo.launch(share=True)
