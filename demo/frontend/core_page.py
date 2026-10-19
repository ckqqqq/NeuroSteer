"""Explain the intervention and render the combined-feature continuation panel."""
import streamlit as st
from frontend_utils import response_request


def main_page(tab_name, tab):
    st.markdown("### From features to interventions")
    st.markdown(
        "Compare positive/negative activation statistics, select SAE features, "
        "decode their directions, and add the weighted directions to GPT-2's layer-6 residual stream. "
        "The sliders combine directions; output effects need not be linear or independent."
    )
    st.subheader("Try a text prefix")
    generation_panel = st.container(height=400)
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Enter an English text prefix to continue."}]
    for message in st.session_state.messages:
        with generation_panel.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Text prefix", key="combined_prompt")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = response_request(prompt, val_info=st.session_state.vals)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # App-level reruns also work during normal widget-triggered script runs.
        st.rerun()
