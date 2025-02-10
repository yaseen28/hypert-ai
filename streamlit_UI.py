import streamlit as st
import time
import re
import pandas as pd
from llama_cpp import Llama

# Page Configuration
st.set_page_config(
    page_title="Pediatric Hypertension Guidelines Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define model paths
MODEL_PATHS = {
    "LoRA": r"C:\Users\Lora.gguf",
    "QLoRA": r"C:\Users\QLORA.gguf",
    "BitFit": r"C:\Users\BitFit.gguf",
    "BaseModel": r"C:\Users\llama.gguf",
}

# Initialize session state variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "BitFit"
if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = []  # Store feedback history

# Sidebar configuration
with st.sidebar:
    st.header("🧠 Model Selection")
    st.session_state.selected_model = st.radio(
        "Choose a model:", list(MODEL_PATHS.keys()),
        index=list(MODEL_PATHS.keys()).index(st.session_state.selected_model)
    )

    if st.button("🗑️ Clear Chat History"):
        st.session_state.conversation_history = []
        st.rerun()


# Load selected GGUF model
@st.cache_resource(show_spinner=True)
def load_model(model_path):
    return Llama(
        model_path=model_path,
        n_ctx=8192,
        n_threads=4,
        verbose=False
    )


# Load the currently selected model
model = load_model(MODEL_PATHS[st.session_state.selected_model])

# Main content area
st.title("🏥 Pediatric Hypertension Guidelines Assistant")
st.markdown("""
### Welcome to the Pediatric Hypertension Guidelines Assistant!
This tool helps clinicians quickly access expert guidelines on pediatric hypertension.
Enter a clinical question, and receive expert-aligned responses.
""")

# Response generation parameters
RESPONSE_CONFIG = {
    "max_tokens": 120,
    "temperature": 0.2,
    "top_p": 0.6,
    "top_k": 30,
    "stop": ["\n\n", "##"],
}


# Input validation
def validate_input(text):
    return len(text.split()) >= 3 and text.lower().strip() not in ["", "na", "n/a"]


# User input field
input_text = st.text_area(
    "📝 Enter your clinical question:",
    height=150,
    placeholder="e.g., What are the first-line treatment options for hypertensive emergency in adolescents?",
    help="Be specific about patient age, condition severity, and comorbidities"
)


# 🛠 **Alpaca-style prompt format**
def format_alpaca_prompt(instruction, input_text):
    # Only include the instruction and input, but not the extra labels for response
    return f"{instruction}: {input_text}\nAnswer:"


# Generate response button
if st.button("💡 Get Expert Response", type="primary"):
    if not validate_input(input_text):
        st.warning("⚠️ Please enter a detailed clinical question (minimum 3 words)")
        st.stop()

    with st.spinner(f"🔍 Analyzing with {st.session_state.selected_model}..."):
        try:
            # Construct prompt using Alpaca format
            prompt = format_alpaca_prompt("", input_text)

            start_time = time.time()
            response = model(
                prompt,
                max_tokens=RESPONSE_CONFIG["max_tokens"],
                temperature=RESPONSE_CONFIG["temperature"],
                top_p=RESPONSE_CONFIG["top_p"],
                top_k=RESPONSE_CONFIG["top_k"]
            )
            response_time = time.time() - start_time

            raw_response = response["choices"][0]["text"].strip()

            # Update conversation history
            st.session_state.conversation_history.append(("You", input_text))
            st.session_state.conversation_history.append(
                ("Assistant", f"**Model:** {st.session_state.selected_model}\n{raw_response}")
            )

            # Store last response for feedback
            st.session_state.last_response = {
                "question": input_text,
                "response": raw_response,
                "model": st.session_state.selected_model,
                "feedback": None
            }

        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            st.stop()

# Display conversation history
st.subheader("🗨️ Conversation History")
for speaker, message in reversed(st.session_state.conversation_history):
    with st.container(border=True):
        if speaker == "You":
            st.markdown(f"**🧑⚕️ You:** {message}")
        else:
            cols = st.columns([1, 20])
            with cols[0]:
                st.markdown("🤖")
            with cols[1]:
                st.markdown(message)

# Display response time
if "response_time" in locals():
    st.success(f"✅ Response generated in {response_time:.2f}s | Model: {st.session_state.selected_model}")

# 🏥 Feedback Section 🏥
st.subheader("💬 Provide Feedback on the Response")
if "last_response" in st.session_state:
    st.write(f"**Question:** {st.session_state.last_response['question']}")
    st.write(f"**Model Response:** {st.session_state.last_response['response']}")

    col1, col2, col3 = st.columns(3)

    # 👍 Button
    with col1:
        if st.button("👍 Accurate", key="feedback_correct"):
            st.session_state.last_response["feedback"] = "Correct"
            st.session_state.feedback_data.append(st.session_state.last_response)
            st.success("✅ Feedback recorded: Response is correct!")

    # 👎 Button
    with col2:
        if st.button("👎 Inaccurate", key="feedback_incorrect"):
            st.session_state.last_response["feedback"] = "Incorrect"
            st.session_state.feedback_data.append(st.session_state.last_response)
            st.warning("⚠️ Feedback recorded: Response is incorrect. Please provide correction.")

    # Textbox for correction
    with col3:
        correction = st.text_area("📝 Suggest a better response", key="correction_input", height=100)
        if st.button("📤 Submit Correction"):
            if correction.strip():
                st.session_state.last_response["corrected_response"] = correction.strip()
                st.session_state.feedback_data.append(st.session_state.last_response)
                st.success("✅ Correction submitted! This will help improve future responses.")
            else:
                st.warning("⚠️ Please enter a corrected response before submitting.")

# 🔄 Save Feedback Data
if st.button("💾 Save Feedback Data"):
    feedback_df = pd.DataFrame(st.session_state.feedback_data)
    feedback_df.to_csv("feedback_data.csv", index=False)
    st.success("✅ Feedback data saved!")

# Footer
st.markdown("""
---
📧 **For feedback & support:** [Contact Us](mailto:support@healthcareassistant.com)  
🔐 Patient data is neither stored nor processed
""")
