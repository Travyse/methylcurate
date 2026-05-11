__all__ = ["help_node"]

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ...utils.prompting import render_prompt


async def help_node(state: Any, config: RunnableConfig) -> dict[str, Any]:
    content = render_prompt("help/help_message.md")
    return {
        "messages": [AIMessage(content=content)],
        "next_action_hint": "Help displayed.",
    }
