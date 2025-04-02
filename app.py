import streamlit as st
from chains import rag_chain

# Streamlit App UI
st.title("Deen Pal Chatbot")

# Chat History State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept User Input
if prompt := st.chat_input("Please type your question"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Store User Query
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Perform Retrieval and Generate Answer
    response = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.messages})

    with st.chat_message("assistant"):
        st.markdown(response["answer"])

    # Store Assistant Response
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})