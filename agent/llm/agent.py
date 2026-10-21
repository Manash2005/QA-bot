from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from agent.tools.rag_tool import search_knowledge_base

load_dotenv()
tools = [search_knowledge_base]

llm = ChatOpenRouter(
    model="openrouter/free",
    temperature=0,
).bind_tools(tools)


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a document-grounded assistant. "
        "Use search_knowledge_base for questions about the user's "
        "documents. Base document-specific claims on retrieved content. "
        "If the documents do not contain enough information, say so. "
        "Do not invent facts or citations. Cite source filenames and "
        "page numbers when available."
    )
)


def call_model(state: MessagesState):
    messages = [SYSTEM_PROMPT, *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}


def route_after_model(state: MessagesState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


graph = StateGraph(MessagesState)

graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")

graph.add_conditional_edges(
    "agent",
    route_after_model,
    ["tools", END],
)

graph.add_edge("tools", "agent")

agent = graph.compile()

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What new school structure does the policy propose to replace the 10+2 structure, and which ages does it cover?",
                }
            ]
        },
        config={"recursion_limit": 10},
    )

    print(result["messages"][-1].content)