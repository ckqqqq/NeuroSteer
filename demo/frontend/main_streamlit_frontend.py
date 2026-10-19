import streamlit as st
from frontend_utils import response_request

st.set_page_config(page_title="NeuroSteer Demo", layout="wide")

# Show the real inference state instead of letting visitors mistake a UI for a ready model.
import os
import requests
backend_base_url = os.getenv("NEUROSTEER_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
try:
    backend_status = requests.get(f"{backend_base_url}/health", timeout=(2, 3))
    if backend_status.status_code != 200:
        st.warning("页面已上线，实时生成暂不可用：模型或四组任务缓存尚未就绪。当前可查看交互界面。")
except requests.RequestException:
    st.warning("页面已上线，推理服务暂未连接。实时生成暂不可用。")

st.caption("GPT-2 文本续写研究演示：滑块系数沿用原始代码，不等同于论文中的归一化 alpha。需要真实模型和四组缓存。")
# 初始化全局聊天历史
targets = ["Positive", "Toxic", "Supportive", "Polite"]
keys = ["sen", "tox", "sta", "pol"]

st.session_state.vals = {"sen": 0, "tox": 0, "sta": 0, "pol": 0}

val_zero = {"sen": 0, "tox": 0, "sta": 0, "pol": 0}
emojis = [
    {"label": "开心", "symbol": "😄|🎉|✨"},
    {"label": "咒骂", "symbol": "🤬|💢|🗯️"},
    {"label": "恶魔", "symbol": "👿|🔥|🔱"},
    {"label": "天使", "symbol": "🕊️|👼|🌟"},
    {"label": "支持（Yes）", "symbol": "👍|✅|🟢"},
    {"label": "反对（No）", "symbol": "👎|❌|🔴"},
    {"label": "礼貌（礼帽）", "symbol": "📜|🎩|🕴️"},
    {"label": "粗鲁（野蛮人）", "symbol": "🤪|🪓|🌪️"},
]


# {"label": "机器", "symbol": "🤖🤖🤖"},
def get_emoji(key, alpha):
    if alpha == 0:
        return "🤖"
    task_index = keys.index(key)
    emoji_index = task_index * 2 + int(alpha < 0)
    return emojis[emoji_index]["symbol"].split("|")[0]


tab_names = ["NeuroSteer"] + targets


# 模型响应生成函数
def tab_response(user_input, target, key, alpha_val, emoji=""):
    # 这里可以替换为实际的模型生成逻辑
    print("Input", user_input, target, key, alpha_val)
    val_info_zero = val_zero.copy()
    val_info_copy = val_zero.copy()
    val_info_copy[key] = alpha_val
    res_A_text = (
        response_request(user_input, val_info_zero).replace("\n", " ").replace("\r", "")
    )
    res_B_text = (
        response_request(user_input, val_info_copy).replace("\n", " ").replace("\r", "")
    )
    emo = f"➕{emoji}" if alpha_val != 0 else ""
    response_A = f"**{res_A_text}**"
    response_B = emo + f"**{res_B_text}**"
    # 更新对应标签页的聊天记录
    print(response_B)
    return response_A, response_B


# 清除历史函数
def clear_history():
    if "tab_id" in st.session_state:
        tab_name = st.session_state.tab_id
        st.session_state[f"messages_a_{tab_name}"] = []
        st.session_state[f"messages_b_{tab_name}"] = []
    st.session_state.messages = [{"role": "assistant", "content": "Hello~"}]


import streamlit as st


st.sidebar.slider(
    "+Positive Sentiment", min_value=-300, max_value=300, value=0, key="sen_value"
)
# st.write('Values:', values)
st.sidebar.slider("+Toxic", -100, 100, value=0, key="tox_value")
st.sidebar.slider("+Supportive", -30, 30, value=0, key="sta_value")
st.sidebar.slider("+Polite", -30, 30, value=0, key="pol_value")

if st.session_state.sen_value:
    st.sidebar.write(f" sen: {st.session_state.sen_value}")
    st.session_state.vals["sen"] = st.session_state.sen_value
    print("情感")
if st.session_state.tox_value:
    st.sidebar.write(f" tox: {st.session_state.tox_value}")
    st.session_state.vals["tox"] = st.session_state.tox_value
    print("毒性")
if st.session_state.sta_value:
    st.sidebar.write(f" sta: {st.session_state.sta_value}")
    st.session_state.vals["sta"] = st.session_state.sta_value
    print("支持")
if st.session_state.pol_value:
    st.sidebar.write(f" pol: {st.session_state.pol_value}")
    st.session_state.vals["pol"] = st.session_state.pol_value
    print("礼貌")

# Using "with" notation
# 创建标签页

tabs = st.tabs(tab_names)
from core_page import main_page

with tabs[0]:
    main_page(tab_names[0], tabs[0])


@st.fragment
def tab_chat(target: str):  # 避免重新渲染别人
    key = dict(zip(targets, keys))[target]
    alpha_val = st.session_state.vals[key]
    emoji = get_emoji(key, alpha_val)
    # 创建两列布局
    if f"messages_a_{target}" not in st.session_state:
        st.session_state[f"messages_a_{target}"] = []
        st.session_state[f"messages_b_{target}"] = []
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("GPT-2 baseline 🤖")
        # st.markdown(f"", unsafe_allow_html=True)
        bot_a_messages = st.container()
        for msg in st.session_state[f"messages_a_{target}"]:
            with bot_a_messages.chat_message(msg["role"]):
                st.markdown(msg["content"])

    with col2:
        head_text = (
            "GPT-2"
            + ("➖" if alpha_val < 0 else "➕")
            + target
            + f" ("
            + str(abs(alpha_val))
            + ")"
        )
        st.subheader(head_text + emoji)
        bot_b_messages = st.container()
        for msg in st.session_state[f"messages_b_{target}"]:
            with bot_b_messages.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Enter text and press ENTER", key=f"input_{target}")
    if user_input:
        st.session_state.tab_id = target
        alpha_val = st.session_state.vals[key]
        print(st.session_state.vals)
        response_A, response_B = tab_response(
            user_input=user_input,
            target=target,
            key=key,
            alpha_val=alpha_val,
            emoji=emoji,
        )
        st.session_state[f"messages_a_{target}"].append(
            {"role": "user", "content": user_input}
        )
        st.session_state[f"messages_b_{target}"].append(
            {"role": "user", "content": user_input}
        )
        st.session_state[f"messages_a_{target}"].append(
            {"role": "assistant", "content": response_A}
        )
        st.session_state[f"messages_b_{target}"].append(
            {"role": "assistant", "content": response_B}
        )
        st.rerun()


for tab_name, tab in zip(tab_names[1:], tabs[1:]):
    with tab:
        tab_chat(target=tab_name)


with st.sidebar:
    if st.button("🗑️ Clear history", key="clear_"):
        clear_history()
        st.rerun()
    st.caption("SAE feature interventions · GPT-2 layer 6 · Research demo")
