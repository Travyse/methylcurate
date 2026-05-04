# api/session.py
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class RunSession:
    """
    Represents a single run session, including its state, task, and any pending interrupts.

    Attributes:
        run_id (str): Unique identifier for the run session.
        queue (asyncio.Queue): Queue for managing messages or tasks related to the session.
        task (Optional[asyncio.Task]): The main task associated with the session, if any.
        main_state (Any): The main state of the session, which can hold any relevant data.
        pending_interrupt (Optional[dict]): A dictionary representing any pending interrupts for the session, if any.
    """

    run_id: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: Optional[asyncio.Task] = None
    main_state: Any = None
    pending_interrupt: Optional[dict] = None


class SessionStore:
    """
    Manages multiple run sessions, allowing creation, retrieval, and deletion of sessions.

    Attributes:
        _sessions (Dict[str, RunSession]): A dictionary mapping run IDs to their corresponding RunSession objects.
    """

    def __init__(self):
        self._sessions: Dict[str, RunSession] = {}

    def create(self, run_id: str) -> RunSession:
        """
        Creates a new run session with the given run ID. If a session with the same run ID already exists, it returns the existing session.

        Args:
            run_id (str): The unique identifier for the run session to be created.

        Returns:
            RunSession: The newly created or existing run session associated with the given run ID.
        """
        if run_id in self._sessions:
            return self._sessions[run_id]
        s = RunSession(run_id=run_id)
        self._sessions[run_id] = s
        return s

    def exists(self, run_id: str) -> bool:
        """
        Checks if a run session with the given run ID exists.

        Args:
            run_id (str): The unique identifier for the run session to check.

        Returns:
            bool: True if a session with the given run ID exists, False otherwise.
        """
        return run_id in self._sessions

    def get(self, run_id: str) -> RunSession:
        """
        Retrieves the run session associated with the given run ID.

        Args:
            run_id (str): The unique identifier for the run session to retrieve.

        Returns:
            RunSession: The run session associated with the given run ID.
        """
        return self._sessions[run_id]

    def delete(self, run_id: str) -> None:
        """
        Deletes the run session associated with the given run ID.

        Args:
            run_id (str): The unique identifier for the run session to delete.
        """
        self._sessions.pop(run_id, None)
