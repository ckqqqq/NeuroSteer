
import streamlit as st
from response_utils import response_generator
def mind_map(tab_name,tab):
    import streamlit.components.v1 as components
    p = open("mind_map.html", encoding="utf-8")
    # st.container()
    with st.container(height=400):
        components.html(p.read(), height=400, scrolling=True)

@st.fragment
def main_bot():
    main_bot = st.container(height=400)
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello~"}]
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
      with main_bot.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Input Here~"):
        with main_bot.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        response = response_generator(prompt,val_info=st.session_state.demo_alpha_info)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun(scope="fragment")
def NeuroSteer(tab_name,tab):
    mind_map(tab_name,tab)
    st.subheader("Try NeuroSteer ↓")
    main_bot()
# Streamed response emulator
