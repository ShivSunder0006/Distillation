import os
import gradio as gr
import time
import random
import plotly.graph_objects as go
import pandas as pd

# Import backend services
from backend.services import groq_service
from backend.services import rag_service

# ----------------------------------------------------
# Visualization Functions (Plotly)
# ----------------------------------------------------

def create_flow_network(alignment_score, t_len, s_len):
    """Creates a static network graph simulating the Distillation pipeline flow."""
    # Nodes: Input -> Teacher, Student Base, CoT -> Distill Loss -> Student Distilled 
    node_labels = ["Input", "Teacher (70b)", "Student (Base 8b)", "Reasoning CoT", "Distillation Loss", "Student (Distilled 8b)"]
    node_x = [0, 2, 2, 4, 6, 8]
    node_y = [1, 2, 0, 2, 1, 1]
    node_colors = ["#e9e6f7", "#6C5CE7", "#00d2ff", "#9e93ff", "#ff6e84", "#00c3ed"]

    edge_x = []
    edge_y = []
    # Input -> Teacher, Input -> Base, Input -> Distill
    # Teacher -> CoT -> Distill Loss -> Distill
    for edge in [(0, 1), (0, 2), (1, 3), (3, 4), (0, 5), (4, 5)]:
        edge_x.extend([node_x[edge[0]], node_x[edge[1]], None])
        edge_y.extend([node_y[edge[0]], node_y[edge[1]], None])
        
    fig = go.Figure()
    
    # Draw edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#474754'),
        hoverinfo='none',
        mode='lines'
    ))

    # Add invisible edge for strokeDash effect on the loss line
    fig.add_trace(go.Scatter(
        x=[node_x[4], node_x[5], None],
        y=[node_y[4], node_y[5], None],
        line=dict(width=3, color='#ff6e84', dash='dash'),
        hoverinfo='none',
        mode='lines'
    ))

    # Draw nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_labels,
        textposition="top center",
        marker=dict(size=30, color=node_colors, line=dict(width=2, color='white')),
        hoverinfo='text',
        textfont=dict(color="#e9e6f7", size=12, family="Inter")
    ))

    fig.update_layout(
        title="Knowledge Distillation Flow",
        title_font=dict(color="#aca3ff", size=18, family="Space Grotesk"),
        showlegend=False,
        plot_bgcolor="#12121e",
        paper_bgcolor="#12121e",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
        height=300
    )
    return fig

def create_alignment_chart(align_score, t_count, s_count):
    """Creates a Bar chart comparing step counts & gauges the alignment."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=['Teacher', 'Student'],
        x=[t_count, s_count],
        orientation='h',
        marker=dict(color=["#6C5CE7", "#00c3ed"]),
        text=[f"{t_count} steps", f"{s_count} steps"],
        textposition='auto',
    ))

    fig.update_layout(
        title=f"Reasoning Compression (Alignment fidelity: {align_score*100:.1f}%)",
        title_font=dict(color="#aca3ff", size=16),
        xaxis_title="Logic Steps",
        plot_bgcolor="#12121e",
        paper_bgcolor="#12121e",
        font=dict(color="#e9e6f7", family="Inter"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=250
    )
    return fig

def create_token_prob_chart(tokens, probs):
    """Line chart for token probabilities."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(tokens))),
        y=probs,
        mode='lines+markers',
        marker=dict(color="#00d2ff", size=8),
        line=dict(color="#00d2ff", width=2),
        text=tokens,
        hovertemplate="Token: '%{text}'<br>Probability: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="Token Probability Map",
        title_font=dict(color="#aca3ff", size=16),
        xaxis_title="Sequence Depth",
        yaxis=dict(title="Probability Mass", range=[0, 1.05]),
        plot_bgcolor="#1e1e2d",
        paper_bgcolor="#12121e",
        font=dict(color="#e9e6f7", family="Inter"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=300
    )
    return fig

# ----------------------------------------------------
# Inference Logic
# ----------------------------------------------------

def run_distillation(question):
    """Runs the 3 models sequentially and returns markdown UI + graphs."""
    if not question:
        empty_flow = create_flow_network(0, 0, 0)
        empty_bar = create_alignment_chart(0, 0, 0)
        return empty_flow, empty_bar, "Please enter a question.", "", ""
    
    t_steps, t_ans, _ = groq_service.generate_teacher_cot(question)
    
    if isinstance(t_steps, list) and len(t_steps) > 0 and isinstance(t_steps[0], dict) and t_steps[0].get("content") == "Groq API key missing":
        empty_flow = create_flow_network(0, 0, 0)
        return empty_flow, empty_flow, "⚠️ **Configuration Error**: `GROQ_API_KEY` is missing from the `.env` file.", "", ""

    s_base = groq_service.generate_student_direct(question)
    s_dist_steps, s_dist_ans, _ = groq_service.generate_student_distilled(question, t_steps)

    # Format teacher steps
    t_text = "### Teacher Reasoning (70B)\n"
    for s in t_steps:
        t_text += f"> **Step {s['step_number']}**: {s['content']}\n\n"
    t_text += f"#### **Final Answer**:\n{t_ans}"

    # Format student base
    s_base_text = f"### Student Direct (8B)\n*(No Chain-of-Thought)*\n\n#### **Final Answer**:\n{s_base}"

    # Format student distilled steps
    s_dist_text = "### Student Distilled (8B)\n"
    for s in s_dist_steps:
        tag = ""
        if s.get("is_missing"):
            tag = " ❌ *(Missing/Compressed Logic)*"
        s_dist_text += f"> **Step {s['step_number']}**: {s['content']}{tag}\n\n"
    s_dist_text += f"#### **Final Answer**:\n{s_dist_ans}"

    # Generate Graphs
    alignment_score = max(0.4, 1.0 - (abs(len(t_steps) - len(s_dist_steps)) * 0.1))
    flow_fig = create_flow_network(alignment_score, len(t_steps), len(s_dist_steps))
    align_fig = create_alignment_chart(alignment_score, len(t_steps), len(s_dist_steps))

    return flow_fig, align_fig, t_text, s_base_text, s_dist_text

def run_experiment(question, temp, top_k, top_p, use_cot):
    if not question:
        empty_plot = create_token_prob_chart([], [])
        return "Please enter a question.", [], empty_plot
    
    output = groq_service.generate_experiment(question, temp, top_k, top_p, use_cot)
    
    # Simulate probabilities mapping
    tokens = output.split()
    highlighted_tokens = []
    raw_probs = []
    
    for token in tokens[:50]: # Limit scale
        prob = min(0.99, max(0.1, random.gauss(0.85, 0.15)))
        raw_probs.append(prob)
        # Highlight mapping [-1, 1]
        score = (prob * 2) - 1.0
        highlighted_tokens.append((token + " ", score))
        
    if len(tokens) > 50:
        highlighted_tokens.append(("... (truncated)", 0.0))
        
    line_fig = create_token_prob_chart(tokens[:50], raw_probs)
        
    return output, highlighted_tokens, line_fig

# RAG Functions
def process_upload(file_path):
    if not file_path:
        return "No file provided."
    try:
        filename = os.path.basename(file_path)
        chunks = rag_service.process_pdf(file_path, filename)
        return f"Successfully processed {chunks} chunks from PDF into FAISS index."
    except Exception as e:
        return f"Error: {str(e)}"

def run_rag_query(message, history):
    if not message:
        return "Empty query."
    try:
        answer, chunks = rag_service.query_pipeline(message)
        source_text = "\n\n---\n**Sources Retrieved:**\n"
        for i, c in enumerate(chunks):
            source_text += f"- *{c['source']} (L2 Distance: {c['score']})*: {c['content'][:150]}...\n"
        return answer + source_text
    except Exception as e:
        return f"System Error: {str(e)}"


# ----------------------------------------------------
# Gradio UI Definition
# ----------------------------------------------------

css = """
body { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
.gradio-container { background-color: #0d0d18 !important; }
.dark .gradio-container { background-color: #0d0d18 !important; }
"""

theme = gr.themes.Monochrome(
    neutral_hue="slate",
    primary_hue="purple",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#0d0d18",
    body_text_color="#e9e6f7",
    background_fill_primary="#12121e",
    background_fill_secondary="#1e1e2d",
    border_color_primary="rgba(172,163,255,0.2)",
    block_background_fill="#12121e",
)

with gr.Blocks(theme=theme, css=css, title="CoT Distillation Visualizer") as demo:
    
    gr.HTML("""
    <div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px 0;">
        <h1 style="color: #aca3ff; font-size: 2.5rem; margin-bottom: 10px;">The Ethereal Logic</h1>
        <p style="color: rgba(233,230,247,0.7); font-size: 1.1rem;">
            Visualize Chain-of-Thought (CoT) reasoning distillation between large Teacher models (70B) and smaller Student models (8B).
        </p>
    </div>
    """)
    
    with gr.Tabs():
        # TAB 1: Dashboard
        with gr.Tab("Distillation Pipeline"):
            with gr.Row():
                question_in = gr.Textbox(
                    label="Input Sequence", 
                    placeholder="E.g., If I have 3 apples and eat 2, how many pears do I have if I started with 5 pears?",
                    scale=4
                )
                distill_btn = gr.Button("Distill & Visualize", variant="primary", scale=1)
                
            # Plotly Graphics
            with gr.Row():
                flow_plot = gr.Plot(label="Network Architecture", value=create_flow_network(0,0,0))
                align_plot = gr.Plot(label="Compression Metrics", value=create_alignment_chart(0,0,0))
                
            with gr.Row():
                with gr.Column():
                    t_out = gr.Markdown(value="Waiting for input...", container=True)
                with gr.Column():
                    sb_out = gr.Markdown(value="Waiting for input...", container=True)
                    sd_out = gr.Markdown(value="Waiting for input...", container=True)
            
            # Action
            distill_btn.click(
                fn=run_distillation,
                inputs=[question_in],
                outputs=[flow_plot, align_plot, t_out, sb_out, sd_out]
            )
            question_in.submit(
                fn=run_distillation,
                inputs=[question_in],
                outputs=[flow_plot, align_plot, t_out, sb_out, sd_out]
            )

        # TAB 2: Laboratory & Tokens
        with gr.Tab("Inference Laboratory"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Decoding Parameters")
                    temp_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature")
                    topk_slider = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-K")
                    topp_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-P")
                    cot_toggle = gr.Checkbox(label="Enable CoT Guidance Prompting", value=True)
                    
                    lab_q_in = gr.Textbox(label="Question", placeholder="Ask anything...")
                    lab_btn = gr.Button("Synthesize & Analyze", variant="secondary")
                    
                with gr.Column(scale=3):
                    gr.Markdown("### 1. Model Output")
                    lab_out_text = gr.Textbox(label="Generated Sequence", lines=4)
                    
                    gr.Markdown("### 2. Token Inference Confidence")
                    token_heatmap = gr.HighlightedText(
                        label="Token Uncertainty Map",
                        color_map={"-1.0": "red", "0.0": "yellow", "1.0": "green"},
                        combine_adjacent=False
                    )
                    
                    gr.Markdown("### 3. Log-Probability Graph")
                    token_line_plot = gr.Plot(value=create_token_prob_chart([], []))
            
            lab_btn.click(
                fn=run_experiment,
                inputs=[lab_q_in, temp_slider, topk_slider, topp_slider, cot_toggle],
                outputs=[lab_out_text, token_heatmap, token_line_plot]
            )

        # TAB 3: RAG
        with gr.Tab("RAG & Grounding"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Context Injection")
                    pdf_input = gr.File(label="Upload Research Paper (PDF)", file_types=[".pdf"])
                    upload_btn = gr.Button("Embed & Index into FAISS")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)
                    
                    upload_btn.click(
                        fn=process_upload,
                        inputs=[pdf_input],
                        outputs=[upload_status]
                    )
                    
                with gr.Column(scale=2):
                    gr.Markdown("### 2. Retrieval Guided Generation")
                    chatbot = gr.ChatInterface(
                        fn=run_rag_query,
                        chatbot=gr.Chatbot(height=450),
                        textbox=gr.Textbox(placeholder="Ask a question about the uploaded document...", scale=7)
                    )

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    # Important: In 4.0+, pass css/theme to gr.Blocks() directly. Share=False for local dev.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
