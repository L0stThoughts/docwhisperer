"""Chat component stub for Streamlit frontend

This module will implement a reusable chat UI component that displays the
conversation history, rendered sources with citations, and supports streaming
LLM responses.

Planned public functions/classes:
- ChatComponent
    - add_user_message(text: str)
    - add_assistant_message(text: str, sources: list)
    - render()
"""
from typing import List, Dict


class ChatComponent:
    """A simple chat component abstraction for the Streamlit app.

    The real implementation will keep session state for messages and provide
    rendering helpers that show sources with clickable links and highlighted
    excerpts.
    """

    def __init__(self) -> None:
        self.messages: List[Dict] = []

    def add_user_message(self, text: str) -> None:
        """Add a user message to the chat history."""
        raise NotImplementedError

    def add_assistant_message(self, text: str, sources: List[Dict]) -> None:
        """Add an assistant message with associated sources/provenance."""
        raise NotImplementedError

    def render(self) -> None:
        """Render the chat to the Streamlit page."""
        raise NotImplementedError
