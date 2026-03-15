# ruff: noqa: BLE001

import contextlib
import logging
from datetime import UTC, datetime, timedelta
from threading import Lock

import redshift_connector
from redshift_connector import Connection

logger = logging.getLogger(__name__)


class RedshiftConnectionPool:
    """Connection pool for Redshift with round-robin access and automatic reset after 50 queries or 30 minutes."""

    def __init__(  # noqa: PLR0913
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 5439,
        pool_size: int = 4,
        reset_after: int = 50,
        max_connection_age_minutes: int = 30,
    ) -> None:
        self._host = host
        self._database = database
        self._user = user
        self._password = password
        self._port = port
        self._pool_size = pool_size
        self._reset_after = reset_after
        self._max_connection_age = timedelta(minutes=max_connection_age_minutes)

        self._connections = []
        self._query_counts = []
        self._connection_created_at = []
        self._current_index = 0
        self._lock = Lock()

        # Initialize all connections at startup
        for _ in range(pool_size):
            conn = self._create_connection()
            self._connections.append(conn)
            self._query_counts.append(0)
            self._connection_created_at.append(datetime.now(tz=UTC))

    def _create_connection(self) -> Connection:
        """Create a new Redshift connection."""
        conn = redshift_connector.connect(
            host=self._host,
            database=self._database,
            user=self._user,
            password=self._password,
            port=self._port,
            tcp_keepalive=True,
            tcp_keepalive_count=3,
            tcp_keepalive_idle=30,
            tcp_keepalive_interval=30,
        )
        conn.autocommit = True
        return conn

    def _reset_connection(self, index: int) -> None:
        """Reset a connection at the given index."""
        try:
            self._connections[index].close()
        except Exception:
            logger.info("Failed to close connection")
            # Ignore errors when closing

        self._connections[index] = self._create_connection()
        self._query_counts[index] = 0
        self._connection_created_at[index] = datetime.now(tz=UTC)

    @property
    def connection(self) -> Connection:
        """Get the next available connection in round-robin fashion."""
        with self._lock:
            current_idx = self._current_index

            # Increment query count
            self._query_counts[current_idx] += 1

            # Calculate connection age
            connection_age = datetime.now(tz=UTC) - self._connection_created_at[current_idx]

            # Check if we need to reset this connection (by query count or age)
            if self._query_counts[current_idx] >= self._reset_after or connection_age >= self._max_connection_age:
                self._reset_connection(current_idx)

            # Get current connection (after potential reset)
            conn = self._connections[current_idx]

            # Validate connection is alive
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            except Exception:
                logger.info("Connection validation failed, resetting")
                self._reset_connection(current_idx)
                conn = self._connections[current_idx]

            # Move to next connection for next request
            self._current_index = (self._current_index + 1) % self._pool_size

            return conn

    def close_all(self) -> None:
        """Close all connections in the pool."""
        for conn in self._connections:
            with contextlib.suppress(Exception):
                conn.close()

    def reset_last_connection(self) -> int:
        """Reset the most recently used connection. Returns the index."""
        with self._lock:
            idx = (self._current_index - 1) % self._pool_size
            self._reset_connection(idx)
            return idx
