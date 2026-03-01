# Thread ID management for multi-turn conversations

import uuid

def new_session() -> str:
    """Generate a new unique session/thread ID."""
    return str(uuid.uuid4())

DEFAULT_SESSION = "default_session"
