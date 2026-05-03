from pydantic import BaseModel
from typing import Any

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