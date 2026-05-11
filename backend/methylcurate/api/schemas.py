from typing import Any

from pydantic import BaseModel


class StartResponse(BaseModel):
    """
    Response model for starting a new run.
    """

    run_id: str


class MessageRequest(BaseModel):
    """
    Request model for sending a message to a run.
    """

    run_id: str
    message: str


class ResumeRequest(BaseModel):
    """
    Request model for resuming a paused run.
    """

    run_id: str
    thread_id: str
    answer: Any


class FilePayload(BaseModel):
    name: str
    content: str
