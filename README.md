# -Task-1-Text-Summarization-with-Transformers-Gradio-
Built during the CodTech Data Science Internship, this task delivers a Python-based NLP tool using BART for abstractive summarization. Integrated with Gradio, it enables instant article compression and showcases skills in model deployment, UI design, and real-world text processing.

python coding starts:
pip install transformers torch gradio
from transformers import pipeline
import gradio as gr

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define summarization function
def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# Optional: Gradio UI for demo
demo = gr.Interface(fn=summarize_text,
                    inputs=gr.Textbox(lines=15, placeholder="Paste your article here..."),
                    outputs="text",
                    title="Text Summarization Tool",
                    description="Summarizes lengthy articles using NLP")

demo.launch()

Direct Task link : https://colab.research.google.com/drive/1JH9DD-Kn-pwYWhcQdzRUXyVcFrOCd4E1?usp=sharing

🧠 Task 1 Summary: Text Summarization Tool In this task, we designed and implemented a Python-based tool that performs abstractive text summarization using state-of-the-art Natural Language Processing (NLP) techniques. The goal was to convert lengthy articles into concise, readable summaries—enhancing accessibility, comprehension, and productivity across domains like education, journalism, and civic tech.

🔍 Approach & Implementation We leveraged the Hugging Face Transformers library and the pretrained facebook/bart-large-cnn model, known for its robust performance in summarizing long-form content. The summarization logic was wrapped in a clean Python function and deployed via a Gradio interface, allowing users to input text and instantly view the generated summary.

Key features:

Abstractive summarization: Generates new sentences rather than extracting existing ones.

Length control: Ensures summaries are concise yet informative.

Interactive UI: Gradio provides a user-friendly demo environment for testing and showcasing.

🧪 Demonstration We tested the tool using a sample article on AI in Education, which highlighted the tool’s ability to distill complex, multi-paragraph content into a brief, coherent summary. The output retained key insights while reducing cognitive load—demonstrating real-world utility.

💡 Insights & Learnings Model selection matters: BART performed well for general-purpose summarization, but task-specific models (e.g., legal, medical) could enhance domain accuracy.

User experience is key: A simple, scrollable interface with clear input/output improves engagement and usability.

Scalability potential: The tool can be modularized for multilingual support, scenario toggles, or integration into larger agentic AI workflows.

✅ Outcome The task is successfully completed with a functional, demo-ready summarization tool. It showcases practical NLP skills, model integration, and UI deployment—making it a strong foundational piece for future projects in civic tech, education, or content automation.
