import asyncio

from langchain_core.messages import HumanMessage


def test_handoff_node_returns_canned_message_and_intent():
    import agent.nodes as nodes
    from agent.prompts import HANDOFF_MESSAGE

    out = asyncio.run(nodes.handoff_node(
        {"messages": [HumanMessage(content="I want a human")], "user_data": {}}
    ))
    assert out["messages"][0].content == HANDOFF_MESSAGE
    assert out["intent"] == "HANDOFF"
