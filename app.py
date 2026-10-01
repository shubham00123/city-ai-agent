from dotenv import load_dotenv
import os
import requests
import streamlit as st

load_dotenv()

from langchain_groq import ChatGroq
from langchain.tools import tool
from tavily import TavilyClient
from langchain.agents import create_agent


# =========================
# 🌦️ Weather Tool
# =========================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    response = requests.get(url, timeout=5)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


# =========================
# 📰 News Tool (Tavily)
# =========================

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=2
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# =========================
# 🧠 LLM Setup
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)


# =========================
# 🤖 Agent
# =========================

agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful city assistant."
)


# =========================
# 🎨 Streamlit UI
# =========================

st.set_page_config(
    page_title="City AI Agent",
    page_icon="🌆"
)

st.title("🌆 City AI Agent")
st.caption("Ask about weather or latest news of any city.")


# =========================
# 💬 Session State
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_tool" not in st.session_state:
    st.session_state.pending_tool = None

if "current_input" not in st.session_state:
    st.session_state.current_input = None

if "tool_queue" not in st.session_state:
    st.session_state.tool_queue = []

if "tool_results" not in st.session_state:
    st.session_state.tool_results = []


# =========================
# 💬 Show Chat History
# =========================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================
# 💬 User Input
# =========================

user_input = st.chat_input(
    "Ask something about a city..."
)


if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    st.session_state.current_input = user_input
    st.session_state.tool_results = []

    try:

        # Ask LLM which tools are required
        response = llm.bind_tools(
            [get_weather, get_news]
        ).invoke(user_input)

        tool_calls = response.tool_calls

        # =========================
        # Tools Required
        # =========================

        if tool_calls:

            st.session_state.tool_queue = tool_calls

            st.session_state.pending_tool = (
                tool_calls[0]["name"]
            )

        # =========================
        # No Tool Required
        # =========================

        else:

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.content
                }
            )

    except Exception as e:

        st.error(f"Error: {e}")

    st.rerun()


# =========================
# 🔧 Tool Approval
# =========================

if st.session_state.pending_tool:

    tool_name = st.session_state.pending_tool

    st.warning(
        f"⚠️ Agent wants to call `{tool_name}`"
    )

    col1, col2 = st.columns(2)


    # =========================
    # ✅ APPROVE
    # =========================

    with col1:

        if st.button("✅ Approve"):

            # Safety check
            if not st.session_state.tool_queue:

                st.session_state.pending_tool = None
                st.rerun()


            # Get first tool
            tool_call = st.session_state.tool_queue.pop(0)

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]


            # =========================
            # Execute Weather
            # =========================

            if tool_name == "get_weather":

                tool_result = get_weather.invoke(
                    tool_args
                )


            # =========================
            # Execute News
            # =========================

            elif tool_name == "get_news":

                tool_result = get_news.invoke(
                    tool_args
                )


            else:

                tool_result = "Unknown tool"


            # =========================
            # Save Tool Result
            # =========================

            st.session_state.tool_results.append(
                f"{tool_name}: {tool_result}"
            )


            # =========================
            # More Tools Remaining
            # =========================

            if st.session_state.tool_queue:

                st.session_state.pending_tool = (
                    st.session_state.tool_queue[0]["name"]
                )


            # =========================
            # All Tools Completed
            # =========================

            else:

                all_results = "\n\n".join(
                    st.session_state.tool_results
                )

                final_prompt = f"""
User asked:
{st.session_state.current_input}

Tool results:
{all_results}

Give a concise and helpful answer to the user.
Use all relevant tool results.
"""

                final_response = llm.invoke(
                    final_prompt
                )


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_response.content
                    }
                )


                # Reset state
                st.session_state.pending_tool = None
                st.session_state.current_input = None
                st.session_state.tool_queue = []
                st.session_state.tool_results = []


            st.rerun()


    # =========================
    # ❌ DENY
    # =========================

    with col2:

        if st.button("❌ Deny"):

            # Safety check
            if st.session_state.tool_queue:

                st.session_state.tool_queue.pop(0)


            # More tools remaining
            if st.session_state.tool_queue:

                st.session_state.pending_tool = (
                    st.session_state.tool_queue[0]["name"]
                )

            else:

                st.session_state.pending_tool = None
                st.session_state.current_input = None
                st.session_state.tool_queue = []
                st.session_state.tool_results = []


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"❌ Tool `{tool_name}` was denied."
                }
            )

            st.rerun()












            