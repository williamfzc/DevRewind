import streamlit as st

from dev_rewind import DevRewind

st.set_page_config(page_title="DevRewind Bot", page_icon=":robot:")


@st.cache_resource
def get_model():
    api = DevRewind()
    agent = api.create_agent()
    return agent


if "messages" not in st.session_state:
    st.session_state.messages = []

for i, (prompt, response) in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)

if prompt := st.chat_input("Start your conversation"):
    model = get_model()
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = model.run(input=prompt)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append((prompt, full_response))
