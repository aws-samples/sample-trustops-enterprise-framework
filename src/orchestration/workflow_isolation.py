"""
Concurrent workflow isolation using workflow_id namespacing.

Prevents cross-workflow interference by isolating workflow state
and resources using workflow_id-based namespacing.

Requirements: 8.13, 8.18
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class IsolatedState:
    """Isolated state for a single workflow.

    Attributes:
        workflow_id: The workflow identifier.
        data: Key-value state data.
        created_at: When the state was created.
        lock: Thread lock for concurrent access.
    """

    workflow_id: str
    data: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from isolated state.

        Args:
            key: The state key.
            default: Default value if key not found.

        Returns:
            The value or default.
        """
        with self.lock:
            return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in isolated state.

        Args:
            key: The state key.
            value: The value to store.
        """
        with self.lock:
            self.data[key] = value

    def delete(self, key: str) -> bool:
        """Delete a key from isolated state.

        Args:
            key: The state key.

        Returns:
            True if the key existed and was deleted.
        """
        with self.lock:
            if key in self.data:
                del self.data[key]
                return True
            return False

    def keys(self) -> list[str]:
        """Get all keys in isolated state."""
        with self.lock:
            return list(self.data.keys())


def namespace_key(workflow_id: str, key: str) -> str:
    """Create a namespaced key for a workflow resource.

    Args:
        workflow_id: The workflow identifier.
        key: The resource key.

    Returns:
        Namespaced key in format ``{workflow_id}/{key}``.
    """
    return f"{workflow_id}/{key}"


def parse_namespace_key(namespaced_key: str) -> tuple[str, str]:
    """Parse a namespaced key into workflow_id and key.

    Args:
        namespaced_key: The namespaced key.

    Returns:
        Tuple of (workflow_id, key).

    Raises:
        ValueError: If the key is not properly namespaced.
    """
    parts = namespaced_key.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid namespaced key: {namespaced_key}"
        )
    return parts[0], parts[1]


class WorkflowIsolation:
    """Manages isolated state for concurrent workflows.

    Uses workflow_id namespacing to prevent cross-workflow
    interference. Each workflow gets its own isolated state
    with thread-safe access.

    Usage::

        isolation = WorkflowIsolation()
        isolation.create_scope("wf-1")
        isolation.set_state("wf-1", "model_id", "m-1")
        value = isolation.get_state("wf-1", "model_id")
    """

    def __init__(self) -> None:
        self._scopes: dict[str, IsolatedState] = {}
        self._global_lock = threading.Lock()

    def create_scope(self, workflow_id: str) -> IsolatedState:
        """Create an isolated scope for a workflow.

        Args:
            workflow_id: The workflow identifier.

        Returns:
            The created IsolatedState.

        Raises:
            ValueError: If scope already exists.
        """
        with self._global_lock:
            if workflow_id in self._scopes:
                raise ValueError(
                    f"Scope already exists: {workflow_id}"
                )
            state = IsolatedState(workflow_id=workflow_id)
            self._scopes[workflow_id] = state
            return state

    def get_scope(
        self, workflow_id: str
    ) -> Optional[IsolatedState]:
        """Get the isolated scope for a workflow.

        Args:
            workflow_id: The workflow identifier.

        Returns:
            The IsolatedState or None if not found.
        """
        with self._global_lock:
            return self._scopes.get(workflow_id)

    def destroy_scope(self, workflow_id: str) -> bool:
        """Destroy an isolated scope and release resources.

        Args:
            workflow_id: The workflow identifier.

        Returns:
            True if the scope existed and was destroyed.
        """
        with self._global_lock:
            if workflow_id in self._scopes:
                del self._scopes[workflow_id]
                return True
            return False

    def set_state(
        self, workflow_id: str, key: str, value: Any
    ) -> None:
        """Set a value in a workflow's isolated state.

        Args:
            workflow_id: The workflow identifier.
            key: The state key.
            value: The value to store.

        Raises:
            KeyError: If the workflow scope does not exist.
        """
        scope = self._get_scope_or_raise(workflow_id)
        scope.set(key, value)

    def get_state(
        self, workflow_id: str, key: str, default: Any = None
    ) -> Any:
        """Get a value from a workflow's isolated state.

        Args:
            workflow_id: The workflow identifier.
            key: The state key.
            default: Default value if key not found.

        Returns:
            The value or default.

        Raises:
            KeyError: If the workflow scope does not exist.
        """
        scope = self._get_scope_or_raise(workflow_id)
        return scope.get(key, default)

    def list_scopes(self) -> list[str]:
        """List all active workflow scopes.

        Returns:
            List of workflow IDs with active scopes.
        """
        with self._global_lock:
            return list(self._scopes.keys())

    def _get_scope_or_raise(self, workflow_id: str) -> IsolatedState:
        with self._global_lock:
            scope = self._scopes.get(workflow_id)
        if scope is None:
            raise KeyError(f"Workflow scope not found: {workflow_id}")
        return scope
