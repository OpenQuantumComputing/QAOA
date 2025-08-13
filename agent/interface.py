import streamlit as st # Framework for building web applications in Python.
# RUN THIS FILE WITH STREAMLIT: `streamlit run interface.py`

import re
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend (prevents popups while running in Streamlit).
import matplotlib.pyplot as plt

# Disable warnings that may show in the Streamlit app.
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="qiskit.visualization.circuit.matplotlib")

from planner import Planner

CODE_FENCE_RE = re.compile(r"```(?P<lang>[^\n]*)\n(?P<code>.*?)```", re.DOTALL) # Regular expression to match fenced code blocks in markdown.

def escape_leading_hashes(text: str) -> str:
    """Escape leading '#' on lines so they don't render as markdown headers.
       Only escapes the first '#' on each line that starts with it.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # If line starts with '#' after stripping left whitespace, escape it.
        if stripped.startswith("#"):
            # Preserve leading indentation, escape the first '#'.
            prefix_len = len(line) - len(stripped)
            lines[i] = line[:prefix_len] + "\\" + line[prefix_len:]
    return "\n".join(lines)

def render_message_content(content: str):
    """Render a message that may contain fenced code blocks and markdown."""
    last = 0
    for m in CODE_FENCE_RE.finditer(content):
        pre = content[last:m.start()]
        if pre.strip():
            st.markdown(escape_leading_hashes(pre))
        lang = m.group("lang").strip() or None
        code = m.group("code").rstrip("\n")
        # Use st.code for code blocks (syntax highlighting supported).
        st.code(code, language=lang)
        last = m.end()
    # Remaining tail.
    tail = content[last:]
    if tail.strip():
        st.markdown(escape_leading_hashes(tail))

# Initialize the planner only once and cache it.
@st.cache_resource
def get_planner():
    return Planner()
planner = get_planner()

if "messages" not in st.session_state: # Initialize chat messages in session state.
    st.session_state.messages = []
if "current_input" not in st.session_state: # Initialize current input in session state.
    st.session_state.current_input = ""
if "pending_response" not in st.session_state: # Initialize pending response flag in session state.
    st.session_state.pending_response = False
if "send_clicked" not in st.session_state: # Initialize send button click state in session state.
    st.session_state.send_clicked = False
# (Runs once when the app starts).

st.title("QAOA Agent Chat") # Chat title.

# If the user has clicked the send button, process the input.
if st.session_state.send_clicked:
    query = st.session_state.current_input
    if query.strip():
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.current_input = ""  # Safe: before widget is rendered. Can try commenting this out.
        st.session_state.pending_response = True
    st.session_state.send_clicked = False
    st.rerun()

# If there is a pending response and messages exist, process the last message.
if st.session_state.pending_response and st.session_state.messages:
    with st.spinner("Thinking..."):
        result = planner(st.session_state.messages[-1]["content"])
    
    captured_media = [] # List to capture media objects (e.g., matplotlib figures, PIL images).
    
    if "```python" in result: # Run Python code blocks and capture media for rendering in chat.
        code = re.search(r"```python\n(.*?)\n```", result, re.DOTALL) # Match Python code blocks.
        if code:
            code_str = code.group(1) # Extract the code block content.
            code_str = re.sub(r"plt\.show\(\)", "", code_str)  # Remove plt.show() calls
            exec_globals = globals()
            exec_locals = {} # Local execution context for the code block.
            
            try:
                matplotlib.use("Agg", force=True) # Force non-interactive backend for matplotlib.
                
                exec(code_str, exec_globals, exec_locals) # Execute the code block.
                seen_figs = set() # Set to track already captured matplotlib figures.
                seen_pil = set() # Set to track already captured PIL images.

                # Active pyplot figure(s):
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    if id(fig) not in seen_figs:
                        captured_media.append(("matplotlib", fig))
                        seen_figs.add(id(fig))

                # Figures/images from exec_locals:
                for value in exec_locals.values():
                    if hasattr(value, "savefig") and hasattr(value, "add_subplot"): # Matplotlib Figure.
                        if id(value) not in seen_figs: # Check if figure is already captured.
                            captured_media.append(("matplotlib", value))
                            seen_figs.add(id(value))
                    elif isinstance(value, Image.Image): # PIL Image.
                        if id(value) not in seen_pil: # Check if PIL image is already captured.
                            captured_media.append(("pil", value))
                            seen_pil.add(id(value))
                            
            except Exception: # Don't crash the app if code execution fails. It this case, the response will just be rendered as usual without output.
                pass

    st.session_state.last_media = captured_media # Store captured media in session state for rendering later.
    st.session_state.messages.append({ # Append the result to the chat messages.
        "role": "assistant",
        "content": result,
        "media": captured_media
    })

    st.session_state.pending_response = False # Reset pending response flag.

# Render chat messages.
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        render_message_content(msg["content"])
        # If this is the last assistant message, attach its respective captured media.
        # This means that media corresponding to previous messages will not be rendered for better performance.
        if msg["role"] == "assistant" and idx == len(st.session_state.messages) - 1:
            if "last_media" in st.session_state:
                for kind, obj in st.session_state.last_media:
                    if kind == "matplotlib":
                        st.pyplot(obj)
                        plt.close(obj)
                    elif kind in ("pil", "bytes"):
                        st.image(obj)

# User input.
query = st.text_area(
    "Your message:", # Label.
    height=150,
    placeholder="Ask me about QAOA...",
    value= st.session_state.current_input,
    key="current_input",
    label_visibility="collapsed" # Hide unnecessary "Your message:" label visually.
)

# Send button to submit the input.
if st.button("Send"):
    st.session_state.send_clicked = True
    st.rerun()