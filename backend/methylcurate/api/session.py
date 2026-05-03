import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

@dataclass
class RunSession:
    """
    Represents a single run session.

    Attributes:
        run_id (str): The unique identifier for the run.
        queue (asyncio.Queue): The queue for managing events in the session.
        main_state (Optional[Any]): The main state of the session.
    """
    run_id: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    main_state: Optional[Any] = None

    # If interrupted, store which thread + payload
    pending_interrupt: Optional[Dict[str, Any]] = None  # {"thread_id": str, "payload": dict}

    # Single active runner task per run
    task: Optional[asyncio.Task] = None

class SessionStore:
    """
    Manages multiple run sessions.

    Attributes:
        _runs (Dict[str, RunSession]): A dictionary mapping run IDs to their corresponding RunSession objects.
    """
    def __init__(self):
        self._runs: Dict[str, RunSession] = {}

    def create(self, run_id: str) -> RunSession:
        """
        Creates a new run session with the given run ID.

        Args:
            run_id (str): The unique identifier for the run.

        Returns:
            RunSession: The newly created run session.
        """
        s = RunSession(run_id=run_id)
        self._runs[run_id] = s
        return s
    
    def list_runs(self) -> List[str]:
        """
        Lists all active run sessions.

        Returns:
            List[str]: A list of run IDs for all active sessions.
        """
        return list(self._runs.keys())

    def get(self, run_id: str) -> RunSession:
        """
        Retrieves a run session by its run ID.

        Args:
            run_id (str): The unique identifier for the run.

        Returns:
            RunSession: The run session associated with the given run ID.
        """
        return self._runs[run_id]

    def exists(self, run_id: str) -> bool:
        """
        Checks if a run session exists for the given run ID.

        Args:
            run_id (str): The unique identifier for the run.

        Returns:
            bool: True if the run session exists, False otherwise.
        """ 
        return run_id in self._runs