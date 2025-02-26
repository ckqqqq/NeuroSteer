import streamlit as st
st.set_page_config(page_title="NeuroSteer DEMO (ACL 2025)", layout="wide")
# 初始化全局聊天历史
targets = ['+positive', '+negative', '+de-toxic', '+toxic', '+polite', '+impolite', '+support', '+oppose']
tab_names = ['NeuroSteer'] + targets
# 模型响应生成函数
def model_response(user_input, tab_key):
    # 这里可以替换为实际的模型生成逻辑
    response_A = f"GPT : {user_input} response"
    response_B = f"GPT{tab_key} : {user_input} response"
    
    # 更新对应标签页的聊天记录
    return response_A,response_B
    

# 清除历史函数
def clear_history():
    tab_name=st.session_state.tab_id
    st.session_state[f"history_A_{tab_name}"] = []
    st.session_state[f"history_B_{tab_name}"] = []
    

import streamlit as st


st.session_state.demo_alpha_info={"sen":0,"tox":0,"sta":0,"pol":0}

st.sidebar.slider(
    '+Positive Sentiment',
    min_value=-300,
    max_value=300, 
    value=0,
    key='sen_value'
)
# st.write('Values:', values)
st.sidebar.slider(
    '+Toxic',
    -100, 100,value=0,
    key='tox_value'
)
st.sidebar.slider(
    '+Supportive', 
    -30, 30, value=0,
    key='sta_value'
)
st.sidebar.slider(
    '+Polite', 
    -30, 30,value=0,
    key='pol_value'
)

if st.session_state.sen_value:
    st.sidebar.write(f" sen: {st.session_state.sen_value}")
    st.session_state.demo_alpha_info["sen"]=st.session_state.sen_value
    print("情感")
if st.session_state.tox_value:
    st.sidebar.write(f" tox: {st.session_state.tox_value}")
    st.session_state.demo_alpha_info["tox"]=st.session_state.tox_value
    print("毒性")
if st.session_state.sta_value:
    st.sidebar.write(f" sta: {st.session_state.sta_value}")
    st.session_state.demo_alpha_info["sta"]=st.session_state.sta_value
    print("支持")
if st.session_state.pol_value:
    st.sidebar.write(f" pol: {st.session_state.pol_value}")
    st.session_state.demo_alpha_info["pol"]=st.session_state.pol_value
    print("礼貌")

# Using "with" notation
# 创建标签页

tabs = st.tabs(tab_names)
from core_page import NeuroSteer
with tabs[0]:
    NeuroSteer(tab_names[0],tabs[0])
    
@st.fragment
def tab_chat(tab_name:str): # 避免重新渲染别人
    # 创建两列布局
    if f"history_A_{tab_name}" not in st.session_state:
        st.session_state[f"history_A_{tab_name}"] =[]
    if f"history_B_{tab_name}" not in st.session_state:
        st.session_state[f"history_B_{tab_name}"] =[]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("GPT")
        bot_a_messages = st.container(height=400)
        for user, bot in st.session_state[f"history_A_{tab_name}"]:
            with bot_a_messages.chat_message("user"):
                    st.write(user)
            with bot_a_messages.chat_message("assistant"):
                    st.write(bot)
    with col2:
        st.subheader(tab_name)
        bot_b_messages = st.container(height=400)
        for user, bot in st.session_state[f"history_B_{tab_name}"]:
            with bot_b_messages.chat_message("user"):
                    st.write(user)
            with bot_b_messages.chat_message("assistant"):
                    st.write(bot)
    user_input=st.chat_input("Enter text and press ENTER",key=f"input_{tab_name}")
    if user_input:
        response_A,response_B=model_response(user_input, tab_name)
        st.session_state[f"history_A_{tab_name}"].append((user_input, response_A))
        st.session_state[f"history_B_{tab_name}"].append((user_input, response_B))
        print("运行",st.session_state[f"history_A_{tab_name}"])
        st.session_state.tab_id=tab_name
        st.rerun(scope="fragment")
    
        
for tab_name,tab in zip(tab_names[1:],tabs[1:]):
    with tab:
        tab_chat(tab_name=tab_name)
        

with st.sidebar:

    cols = st.columns(2)
    with cols[0]:
        if st.button("🗑️ Clear history", key="clear_"):
            clear_history()
            st.rerun()
    with cols[1]:
        if st.button("🗑️ About Us", key="about us"):
            # model_response(user_input, tab_key)
            st.rerun()