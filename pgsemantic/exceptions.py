"""Typed exceptions for pgsemantic.

All third-party exceptions (psycopg.Error, openai.APIError) must be caught
and re-raised as one of these typed exceptions with a human-readable message.
Never let a raw third-party exception propagate to CLI output or MCP responses.
"""


class PgvectorSetupError(Exception):
    """Base exception for all pgsemantic errors."""


class ExtensionNotFoundError(PgvectorSetupError):
    """pgvector extension is not installed or not accessible."""


class TableNotFoundError(PgvectorSetupError):
    """The specified table does not exist in the database."""


class TableNotWatchedError(PgvectorSetupError):
    """The table exists but has not been set up with pgsemantic apply."""


class ColumnNotFoundError(PgvectorSetupError):
    """The specified column does not exist on the table."""


class ColumnAlreadyExistsError(PgvectorSetupError):
    """The embedding column already exists on the table."""


class DimensionMismatchError(PgvectorSetupError):
    """The existing embedding column has different dimensions than the requested model."""


class EmbeddingProviderError(PgvectorSetupError):
    """Error from the embedding model (API error, model not found, etc.)."""


class QueueError(PgvectorSetupError):
    """Error interacting with the pgvector_setup_queue table."""


class ConfigError(PgvectorSetupError):
    """Error reading or writing .pgsemantic.json."""


class MigrationError(PgvectorSetupError):
    """Error during zero-downtime model migration."""


class InvalidFilterError(PgvectorSetupError):
    """A filter key in hybrid_search does not correspond to a valid column."""
