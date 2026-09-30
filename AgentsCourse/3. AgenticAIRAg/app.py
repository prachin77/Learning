"""Run the gala assistant with guest, web search, weather, and Hub tools."""

import os
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from retriever import guest_info_tool
from tools import hub_stats_tool, search_tool, weather_info_tool


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_agent():
    """Configure and compile the LangGraph assistant."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) before starting the app."
        )

    endpoint = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        huggingfacehub_api_token=token,
    )
    chat_with_tools = ChatHuggingFace(llm=endpoint).bind_tools(
        [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]
    )

    def assistant(state: AgentState) -> dict[str, list[AnyMessage]]:
        return {"messages": [chat_with_tools.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node(
        "tools",
        ToolNode([guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]),
    )
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()


def main() -> None:
    agent = build_agent()
    messages: list[AnyMessage] = []
    print("Gala assistant ready. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        print(f"Alfred: {messages[-1].content}")


if __name__ == "__main__":
    main()