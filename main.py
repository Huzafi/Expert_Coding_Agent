import os
import streamlit as st
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

import nest_asyncio
nest_asyncio.apply()

# ----------------- Agent Setup -----------------
gemini_api_key = st.secrets["GEMINI_API_KEY"]

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

coding_agent = Agent(
    name="👨‍💻 Expert Coding Agent",
    instructions="""
        You are an expert coding assistant. 
        Your job is to fix bugs, improve code quality, and explain your fixes.
        Supported languages: Python, JavaScript, C++.
        - Always return corrected code inside proper code blocks.
        - Briefly explain what changes you made and why.
        - Be precise, clear, and helpful like a mentor.
    """
)

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="👨‍💻 Expert Coding Agent", page_icon="👨‍💻")
st.title("👨‍💻 Expert Coding Agent")
st.markdown("""
Welcome! I can help you **debug**, **fix**, and **improve** your code.  
🔧 Just paste your code and tell me what to fix or improve!
""")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Paste your code or describe your bug here...")

async def run_agent_async(user_input: str):
    # Safe async call for Streamlit
    return await Runner.run(
        coding_agent,
        input=user_input,
        run_config=config
    )

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("⏳ Debugging your code...")

    try:
        # Run agent asynchronously but safe in Streamlit
        result = asyncio.run(run_agent_async(user_input))
        response_text = result.final_output

        response_placeholder.markdown(f"✅ **Here’s your fixed code:**\n\n{response_text}")
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    except Exception as e:
        response_placeholder.markdown(f"❌ Error: {str(e)}")

