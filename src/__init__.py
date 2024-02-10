"""
This module serves as the entry point for the trader dashboard application.

It provides access to the CVIngestionPipeline class for data ingestion,
as well as the used_cols and macro_cols variables for column references.

Additionally, it exposes functions for managing the database, including
adding workers, creating and dropping tables for workers and jobs.

"""

from .ingestion import CVIngestionPipeline, used_cols, macro_cols
from .db import (
    add_worker, drop_workers_table, drop_jobs_table, create_jobs_table,
    create_workers_table
)
