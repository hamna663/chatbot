import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Role-Based Chatbot",
    page_icon="🤖",
    layout="centered",
)

# Initialize the model
@st.cache_resource
def load_model():
    return ChatGroq(model="openai/gpt-oss-120b")


model = load_model()

# Prompt template
prompt = PromptTemplate.from_template(
    "You are a {role} assistant. "
    "Respond to the user input accordingly.\n\n"
    "User: {input}\n"
    "Assistant:"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "role" not in st.session_state:
    st.session_state.role = "friendly"

# UI
st.title("🤖 AI Role-Based Chatbot")
st.caption("Chat with an AI assistant and customize its personality.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    role = st.selectbox(
        "Choose assistant personality",
        ["friendly", "sad", "angry"],
        index=["friendly", "sad", "angry"].index(
            st.session_state.role
        ),
    )

    if role != st.session_state.role:
        st.session_state.role = role
        st.session_state.messages = []
        st.rerun()

    st.info(f"Current personality: **{st.session_state.role.title()}**")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Display and save user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                formatted_prompt = prompt.format(
                    role=st.session_state.role,
                    input=user_input,
                )

                response = model.invoke(formatted_prompt)
                answer = response.content

                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(f"Something went wrong: {e}")