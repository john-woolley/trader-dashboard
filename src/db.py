"""
This module contains functions for interacting with a PostgreSQL database in
the context of a trader dashboard application.
It provides functions for creating tables, inserting data,
retrieving data, and managing jobs and workers.
"""

import sqlalchemy as sa
import psycopg2
import pandas as pd
import polars as pl
import polars.selectors as cs
import cloudpickle

from sqlalchemy.sql import update
from sqlalchemy import create_engine, MetaData, Table, Column

from sklearn.decomposition import PCA

CONN = "postgresql+psycopg2://trader_dashboard@0.0.0.0:5432/trader_dashboard"


class DBConnection:
    @staticmethod
    def getconn():
        """
        Returns a connection to the trader_dashboard database.
        """
        c = psycopg2.connect(
            user="trader_dashboard", host="127.0.0.1", dbname="trader_dashboard"
        )
        return c


class RawData:
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)

    @classmethod
    def insert(cls, df: pl.DataFrame, table_name: str) -> None:
        """
        Inserts a pandas DataFrame into a SQL table.

        Args:
            df (pd.DataFrame): The DataFrame to be inserted.
            table_name (str): The name of the SQL table.

        Returns:
            None
        """
        df.write_database(
            table_name,
            CONN,
            if_table_exists="replace",
        )

    @classmethod
    def get(cls, table_name: str) -> pl.LazyFrame:
        """
        Retrieves a raw table from the database.

        Args:
            table_name (str): The name of the table to retrieve.

        Returns:
            pd.DataFrame: The raw table data as a pandas DataFrame.
        """
        table = sa.Table(table_name, cls.metadata, autoload_with=cls.conn)
        query = sa.select(table)
        df = pl.read_database(query, cls.conn)
        return df.lazy()

    @classmethod
    def chunk(cls, table_name: str, chunk_size: int = 1000, no_chunks: int = 0) -> list:
        """
        Chunk the raw table data into smaller chunks and insert them into the
        database table.

        Args:
            table_name (str): The name of the database table.
            chunk_size (int): The size of each chunk.

        Returns:
            list: The list of chunks.
        """
        df = cls.get(table_name)
        df = df.sort("date", "ticker")
        if not no_chunks:
            chunks = cls._chunk_df_by_size(df, chunk_size)
        else:
            chunks = cls._chunk_df_by_number(df, no_chunks)
        cls.insert_chunked_table(chunks, table_name)
        return chunks

    @staticmethod
    def _chunk_df_by_size(df: pl.LazyFrame, chunk_size: int) -> list:
        """
        Chunk a DataFrame into smaller DataFrames of a specified size.

        Args:
            df (pd.DataFrame): The DataFrame to be chunked.
            chunk_size (int): The size of each chunk.

        Returns:
            list: The list of chunks.
        """
        dates = df.select("date").unique().collect().to_series()
        chunks = []
        for i in range(0, len(dates), chunk_size):
            chunk = df.filter(pl.col("date").is_in(dates[i : i + chunk_size]))
            chunks.append(chunk)
        return chunks

    @staticmethod
    def _chunk_df_by_number(df: pl.LazyFrame, no_chunks: int) -> list:
        """
        Chunk a DataFrame into a specified number of chunks.

        Args:
            df (pd.DataFrame): The DataFrame to be chunked.
            no_chunks (int): The number of chunks.

        Returns:
            list: The list of chunks.
        """
        dates = df.select("date").unique().collect().to_series()
        chunk_size = len(dates) // no_chunks
        chunks = []
        for i in range(0, len(dates), chunk_size):
            chunk = df.filter(pl.col("date").is_in(dates[i : i + chunk_size]))
            chunks.append(chunk)
        return chunks

    @classmethod
    def insert_chunked_table(cls, chunks: list, table_name: str) -> None:
        """
        Inserts a list of chunks into a table in the database.

        Args:
            chunks (list): A list of chunks to be inserted.
            table_name (str): The name of the table in the database.

        Returns:
            None
        """
        for i, chunk in enumerate(chunks):
            blob = cloudpickle.dumps(chunk.collect())
            CVData.add(table_name, blob, i)


class Jobs:
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    try:
        table = sa.Table("jobs", metadata, autoload_with=conn)
    except sa.exc.NoSuchTableError:
        table = sa.Table(
            "jobs",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("name", sa.String),
            sa.Column("status", sa.String),
        )

    @classmethod
    def create(cls):
        """
        Creates a jobs table in the database.

        This function connects to the database, creates a metadata object,
        reflects the existing tables,
        defines the structure of the jobs table, and creates the table if it
        doesn't already exist.

        Returns:
            None
        """
        cls.table.create(cls.conn, checkfirst=True)
        cls.conn.commit()

    @classmethod
    def add(cls, job_name: str) -> None:
        """
        Add a job to the database.

        Parameters:
        - job_name (str): The name of the job to add.

        Returns:
        - None
        """
        ins = cls.table.insert().values(name=job_name, status="idle")
        cls.conn.execute(ins)
        cls.conn.commit()

    @classmethod
    def drop(cls):
        """
        Drops the 'jobs' table from the database.
        """
        cls.table.drop(cls.conn, checkfirst=True)
        cls.conn.commit()


class Workers:
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    try:
        jobs_table = sa.Table("jobs", metadata, autoload_with=conn)
    except sa.exc.NoSuchTableError:
        jobs_table = sa.Table(
            "jobs",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("name", sa.String),
            sa.Column("status", sa.String),
        )
    try:
        table = sa.Table("workers", metadata, autoload_with=conn)
    except sa.exc.NoSuchTableError:
        table = sa.Table(
            "workers",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "job_id", sa.Integer, sa.ForeignKey(jobs_table.c.id, ondelete="CASCADE")
            ),
            sa.Column("name", sa.String),
            sa.Column("status", sa.String),
        )

    @classmethod
    def create(cls):
        """
        Create the 'workers' table in the database.

        This function connects to the database, reflects the existing tables,
        and creates a new table called 'workers'.
        The 'workers' table has the following columns:
        - id: Integer, primary key, auto-incremented
        - job_id: Integer, foreign key referencing the 'id' column of the 'jobs'
        table, with CASCADE delete behavior
        - name: String
        - status: String

        After creating the table, the function commits the changes and closes
        the connection.

        """

        cls.table.create(cls.conn, checkfirst=True)
        cls.conn.commit()

    @classmethod
    def add(cls, worker_name: str, jobname: str) -> None:
        """
        Add a worker to the database with the specified worker name and job name.

        Args:
            worker_name (str): The name of the worker.
            jobname (str): The name of the job.

        Returns:
            None
        """
        job_query = sa.select(cls.jobs_table.c.id).where(
            cls.jobs_table.c.name == jobname
        )
        fetched = cls.conn.execute(job_query).fetchone()
        assert fetched is not None, f"Job {jobname} not found"
        job_id = fetched[0]
        worker_table = sa.Table("workers", cls.metadata)
        ins = worker_table.insert().values(
            name=worker_name, status="idle", job_id=job_id
        )
        cls.conn.execute(ins)
        cls.conn.commit()

    @classmethod
    def get_workers_by_name(cls, job_name: str) -> str:
        """
        Retrieve the names of workers associated with a specific job name.

        Args:
            job_name (str): The name of the job.

        Returns:
            str: A comma-separated string of worker names.

        """
        conn = cls.conn
        workers_table = cls.table
        jobs_table = cls.jobs_table
        worker_query = (
            sa.select(workers_table.c.name)
            .select_from(
                workers_table.join(
                    jobs_table, jobs_table.c.id == workers_table.c.job_id
                )
            )
            .where(jobs_table.c.name == job_name)
        )
        res = conn.execute(worker_query).fetchall()
        worker_list = list(map(lambda x: x[0], res))
        workers = ",".join(worker_list)
        return workers

    @staticmethod
    def get_workers_by_id(job_id: int) -> str:
        """
        Retrieve the names of workers associated with a given job ID.

        Args:
            job_id (int): The ID of the job.

        Returns:
            str: A comma-separated string of worker names.

        """
        conn = sa.create_engine(CONN).connect()
        metadata = sa.MetaData()
        metadata.reflect(bind=conn)
        jobs_table = sa.Table("jobs", metadata, autoload_with=conn)
        workers_table = sa.Table("workers", metadata, autoload_with=conn)
        worker_query = (
            sa.select(workers_table.c.name)
            .select_from(
                workers_table.join(
                    jobs_table, jobs_table.c.id == workers_table.c.job_id
                )
            )
            .where(jobs_table.c.id == job_id)
        )
        res = conn.execute(worker_query).fetchall()
        worker_list = list(map(lambda x: x[0], res))
        workers = ",".join(worker_list)
        return workers

    @classmethod
    def drop(cls):
        """
        Drops the 'workers' table from the database.
        """
        cls.table.drop(cls.conn, checkfirst=True)
        cls.conn.commit()


class CVData:
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    try:
        table = sa.Table("cv_data", metadata, autoload_with=conn)
    except sa.exc.NoSuchTableError:
        table = sa.Table(
            "cv_data",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("table_name", sa.String),
            sa.Column("chunk", sa.Integer),
            sa.Column("data", sa.LargeBinary),
        )

    @classmethod
    def create(cls):
        """
        Create a table named 'cv_data' in the database with the specified columns.

        The table will have the following columns:
        - id: Integer, primary key, auto-incremented
        - table_name: String
        - chunk: Integer
        - data: LargeBinary

        This function uses SQLAlchemy to create the table and commits the changes
        to the database.

        Returns:
        None
        """
        cls.table.create(cls.conn, checkfirst=True)
        cls.conn.commit()

    @classmethod
    def add(cls, table_name: str, data: bytes, i: int) -> None:
        """
        Add CV data to the specified table.

        Args:
            table_name (str): The name of the table to add the data to.
            data (bytes): The CV data to be added.
            i (int): The chunk index of the data.

        Returns:
            None
        """
        ins = cls.table.insert().values(table_name=table_name, data=data, chunk=i)
        cls.conn.execute(ins)
        cls.conn.commit()

    @classmethod
    def drop(cls):
        """
        Drops the 'cv_data' table from the database.

        This function connects to the database, retrieves the 'cv_data' table,
        and drops it.
        If the table does not exist, it does nothing.
        """
        cls.table.drop(cls.conn, checkfirst=True)
        cls.conn.commit()

    @classmethod
    def get(cls, table_name: str, i: int) -> bytes:
        """
        Retrieve CV data by table name and chunk index.

        Args:
            table_name (str): The name of the table.
            i (int): The chunk index.

        Returns:
            bytes: The CV data.

        Raises:
            AssertionError: If data is not found for the given table name and
            chunk index.
        """
        table = cls.table
        query = (
            sa.select(table.c.data)
            .where(table.c.table_name == table_name)
            .where(table.c.chunk == i)
        )
        fetched = cls.conn.execute(query).fetchone()
        assert fetched is not None, f"Data not found for {table_name} chunk {i}"
        res = fetched[0]
        return res

    @classmethod
    def get_table_names(cls) -> list:
        """
        Get the names of all tables in the cv_data table.

        Returns:
            list: A list of table names.
        """
        query = sa.select(cls.table.c.table_name).distinct()
        res = cls.conn.execute(query).fetchall()
        names = list(map(lambda x: x[0], res))
        return names

    @classmethod
    def read(cls, table_name: str, i: int) -> pl.LazyFrame:
        """
        Read a chunked table from the database.

        Args:
            table_name (str): The name of the table.
            i (int): The index of the chunk.

        Returns:
            pd.DataFrame: The chunked table as a pandas DataFrame.
        """
        blob = cls.get(table_name, i)
        df = cloudpickle.loads(blob)
        return df.lazy()

    @classmethod
    def standardize(cls, table_name: str, chunk: int) -> None:
        """
        Standardize the columns of a chunked table in the database.

        Args:
            table_name (str): The name of the table.
            chunk (int): The index of the chunk.

        Returns:
            None
        """
        df = cls.read(table_name, chunk).collect()
        if chunk == 0:
            std_df = cls.read(table_name, 0).collect()
        else:
            std_df = pl.concat(
                [cls.read(table_name, i) for i in range(chunk)]
            ).collect()
        preserved_cols = ["open", "high", "low", "close", "closeadj"]
        numerical_cols = [
            col
            for col in df.columns
            if col not in preserved_cols and df[col].dtype.is_numeric()
        ]
        df = df.with_columns(
            (pl.col(col).sub(std_df[col].mean())).truediv(std_df[col].std()).alias(col)
            for col in numerical_cols
        )
        df = df.sort("date", "ticker")
        blob = cloudpickle.dumps(df)
        StdCVData.insert(table_name, chunk, blob)


class StdCVData:
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    try:
        table = sa.Table("std_cv_data", metadata, autoload_with=conn)
    except sa.exc.NoSuchTableError:
        table = sa.Table(
            "std_cv_data",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("table_name", sa.String),
            sa.Column("chunk", sa.Integer),
            sa.Column("data", sa.LargeBinary),
        )

    @classmethod
    def create(cls) -> None:
        """
        Create a standardized cv_data table in the database.

        Returns:
            None
        """
        cls.table.create(cls.conn, checkfirst=True)
        cls.conn.commit()

    @classmethod
    def insert(cls, table_name: str, chunk: int, data: bytes) -> None:
        """
        Insert standardized CV data into the database.

        Args:
            table_name (str): The name of the table.
            chunk (int): The index of the chunk.
            data (bytes): The standardized CV data.

        Returns:
            None
        """
        ins = cls.table.insert().values(table_name=table_name, data=data, chunk=chunk)
        cls.conn.execute(ins)
        cls.conn.commit()

    @classmethod
    def drop(cls):
        """
        Drop the standardized cv_data table from the database.

        Returns:
            None
        """
        cls.table.drop(cls.conn, checkfirst=True)
        cls.conn.commit()

    @classmethod
    def read(cls, table_name: str, chunk: int) -> pl.LazyFrame:
        """
        Read a standardized cv table from the database.

        Args:
            table_name (str): The name of the table.
            chunk (int): The index of the chunk.

        Returns:
            pd.DataFrame: The standardized cv table as a pandas DataFrame.
        """
        blob = cls.get(table_name, chunk)
        df = cloudpickle.loads(blob)
        return df.lazy()

    @classmethod
    def get(cls, table_name: str, chunk: int) -> bytes:
        """
        Retrieve standardized CV data by table name and chunk index.

        Args:
            table_name (str): The name of the table.
            chunk (int): The chunk index.

        Returns:
            bytes: The standardized CV data.
        """
        table = cls.table
        query = (
            sa.select(table.c.data)
            .where(table.c.table_name == table_name)
            .where(table.c.chunk == chunk)
        )
        fetched = cls.conn.execute(query).fetchone()
        assert fetched is not None, f"Data not found for {table_name} chunk {chunk}"
        res = fetched[0]
        return res

    @classmethod
    def calculate_pca_weights(cls, table_name: str, chunk: int) -> bytes:
        """
        Calculate PCA weights for a standardized CV table.

        Args:
            table_name (str): The name of the table.
            chunk (int): The index of the chunk.

        Returns:
            bytes: The PCA weights.
        """
        if chunk == 0:
            df = cls.read(table_name, 0).collect()
        else:
            df = pl.concat([cls.read(table_name, i) for i in range(chunk)]).collect()
        df = df.to_pandas()
        df["momentum"] = df["closeadj"].pct_change().dropna() * df["volume"]
        pivot = df.pivot_table(
            index="date", columns="ticker", values="momentum"
        ).fillna(0)
        pca = PCA(n_components=5)
        pca.fit(pivot)
        weights = pca.components_
        labeled = [f"PC{i}" for i in range(1, 6)]
        weights_df = pd.DataFrame(weights, columns=pivot.columns, index=labeled)
        weights_df = pl.DataFrame(weights_df.T.reset_index())
        blob = cloudpickle.dumps(weights_df)
        PCAWeights.add(table_name, blob, chunk)
        return blob


class PCAWeights:
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    try:
        table = sa.Table("pca_weights", metadata, autoload_with=conn)
    except sa.exc.NoSuchTableError:
        table = sa.Table(
            "pca_weights",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("table_name", sa.String),
            sa.Column("chunk", sa.Integer),
            sa.Column("data", sa.LargeBinary),
        )

    @classmethod
    def create(cls):
        """
        Create a table named 'pca_weights' in the database with the specified columns.

        The table will have the following columns:
        - id: Integer, primary key, auto-incremented
        - table_name: String
        - chunk: Integer
        - data: LargeBinary

        This function uses SQLAlchemy to create the table and commits the changes
        to the database.

        Returns:
        None
        """
        cls.table.create(cls.conn, checkfirst=True)
        cls.conn.commit()

    @classmethod
    def add(cls, table_name: str, data: bytes, i: int) -> None:
        """
        Add PCA weights to the specified table.

        Args:
            table_name (str): The name of the table to add the data to.
            data (bytes): The PCA weights to be added.
            i (int): The chunk index of the data.

        Returns:
            None
        """
        ins = cls.table.insert().values(table_name=table_name, data=data, chunk=i)
        cls.conn.execute(ins)
        cls.conn.commit()

    @classmethod
    def drop(cls):
        """
        Drops the 'pca_weights' table from the database.

        This function connects to the database, retrieves the 'pca_weights' table,
        and drops it.
        If the table does not exist, it does nothing.
        """
        cls.table.drop(cls.conn, checkfirst=True)
        cls.conn.commit()

    @classmethod
    def get(cls, table_name: str, i: int) -> bytes:
        """
        Retrieve PCA weights by table name and chunk index.

        Args:
            table_name (str): The name of the table.
            i (int): The chunk index.

        Returns:
            bytes: The PCA weights.

        Raises:
            AssertionError
        """
        table = cls.table
        query = (
            sa.select(table.c.data)
            .where(table.c.table_name == table_name)
            .where(table.c.chunk == i)
        )
        fetched = cls.conn.execute(query).fetchone()
        assert fetched is not None, f"Data not found for {table_name} chunk {i}"
        res = fetched[0]
        return res

    @classmethod
    def read(cls, table_name: str, i: int) -> pl.DataFrame:
        """
        Read a chunked table from the database.

        Args:
            table_name (str): The name of the table.
            i (int): The index of the chunk.

        Returns:
            pd.DataFrame: The chunked table as a pandas DataFrame.
        """
        blob = cls.get(table_name, i)
        df = cloudpickle.loads(blob)
        return df
    

class RenderData:
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    try:
        table = sa.Table("render_data", metadata, autoload_with=conn)
    except sa.exc.NoSuchTableError:
        table = sa.Table(
            "render_data",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("table_name", sa.String),
            sa.Column("chunk", sa.Integer),
            sa.Column("data", sa.LargeBinary),
        )

    @classmethod
    def create(cls):
        """
        Create a table named 'render_data' in the database with the specified columns.

        The table will have the following columns:
        - id: Integer, primary key, auto-incremented
        - table_name: String
        - chunk: Integer
        - data: LargeBinary

        This function uses SQLAlchemy to create the table and commits the changes
        to the database.

        Returns:
        None
        """
        cls.table.create(cls.conn, checkfirst=True)
        cls.conn.commit()

    @classmethod
    def add(cls, table_name: str, data: pl.DataFrame, i: int) -> None:
        """
        Add render data to the specified table.

        Args:
            table_name (str): The name of the table to add the data to.
            data (bytes): The render data to be added.
            i (int): The chunk index of the data.

        Returns:
        None
        """
        blob = cloudpickle.dumps(data)
        ins = cls.table.insert().values(table_name=table_name, data=blob, chunk=i)
        cls.conn.execute(ins)
        cls.conn.commit()

    @classmethod
    def drop(cls):
        """
        Drops the 'render_data' table from the database.

        This function connects to the database, retrieves the 'render_data' table,
        and drops it.
        If the table does not exist, it does nothing.
        """
        cls.table.drop(cls.conn, checkfirst=True)
        cls.conn.commit()

    @classmethod
    def get(cls, table_name: str, i: int) -> bytes:
        """
        Retrieve render data by table name and chunk index.

        Args:
            table_name (str): The name of the table.
            i (int): The chunk index.

        Returns:
            bytes: The render data.

        Raises:
            AssertionError
        """
        table = cls.table
        query = (
            sa.select(table.c.data)
            .where(table.c.table_name == table_name)
            .where(table.c.chunk == i)
        )
        fetched = cls.conn.execute(query).fetchone()
        assert fetched is not None, f"Data not found for {table_name} chunk {i}"
        res = fetched[0]
        return res
    
    @classmethod
    def read(cls, table_name: str, i: int) -> pl.DataFrame:
        """
        Read a chunked table from the database.

        Args:
            table_name (str): The name of the table.
            i (int): The index of the chunk.

        Returns:
            pd.DataFrame: The chunked table as a pandas DataFrame.
        """
        blob = cls.get(table_name, i)
        df = cloudpickle.loads(blob)
        return df
    



def remove_table_name_from_cv_table(table_name: str) -> None:
    """
    Remove a table from the cv_data table.

    Args:
        table_name (str): The name of the table to remove.

    Returns:
        None
    """
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    table = sa.Table("cv_data", metadata, autoload_with=conn)
    query = sa.delete(table).where(table.c.table_name == table_name)
    conn.execute(query)
    conn.commit()
    conn.close()


def get_workers():
    """
    Get all workers from the database.

    Returns:
        list: A list of workers.
    """
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    workers_table = sa.Table("workers", metadata, autoload_with=conn)
    query = sa.select(workers_table.c.name)
    res = conn.execute(query).fetchall()
    workers = list(map(lambda x: x[0], res))
    conn.close()
    return workers


def get_jobs():
    """
    Get all jobs from the database.

    Returns:
        list: A list of jobs.
    """
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    jobs_table = sa.Table("jobs", metadata, autoload_with=conn)
    query = sa.select(jobs_table.c.name)
    res = conn.execute(query).fetchall()
    jobs = list(map(lambda x: x[0], res))
    conn.close()
    return jobs


def get_job_worker_mapping():
    """
    Get a mapping of jobs to workers.

    Returns:
        dict: A dictionary mapping jobs to workers.
    """
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    jobs_table = sa.Table("jobs", metadata, autoload_with=conn)
    workers_table = sa.Table("workers", metadata, autoload_with=conn)
    query = sa.select(jobs_table.c.name, workers_table.c.name).select_from(
        workers_table.join(jobs_table, jobs_table.c.id == workers_table.c.job_id)
    )
    res = conn.execute(query).fetchall()
    mapping = {}
    for job, worker in res:
        if job not in mapping:
            mapping[job] = [worker]
        else:
            mapping[job].append(worker)
    conn.close()
    return mapping


def get_cv_no_chunks(table_name: str):
    """
    Get the number of chunks for a given table.

    Args:
        table_name (str): The name of the table.

    Returns:
        int: The number of chunks.
    """
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    table = sa.Table("cv_data", metadata)
    query = sa.select(table.c.chunk).where(table.c.table_name == table_name)
    res = conn.execute(query).fetchall()
    conn.close()
    return len(res)


if __name__ == "__main__":
    # Workers.drop()
    # Jobs.drop()
    # CVData.drop()
    # StdCVData.drop()
    # Jobs.create()
    # Workers.create()
    # CVData.create()
    # Jobs.add("test")
    # Workers.add("worker1", "test")
    # Workers.add("worker2", "test")
    # print(Workers.get_workers_by_name("test"))
    # print(Workers.get_workers_by_id(1))
    # test_df = (
    #     pl.read_csv("trader-dashboard/data/master.csv", try_parse_dates=True)
    #     .with_columns(pl.col("date").cast(pl.Date).alias("date"))
    #     .drop("")
    #     .sort("date", "ticker")
    # )
    # test_df = test_df.select(cs.all().forward_fill())
    # RawData.insert(test_df, "test_table")
    # print(RawData.get("test_table"))
    # RawData.chunk("test_table", 240)
    # print(CVData.read("test_table", 0))
    # StdCVData.create()
    # CVData.standardize("test_table", 0)
    # CVData.standardize("test_table", 1)
    # CVData.standardize("test_table", 2)
    # CVData.standardize("test_table", 3)
    # CVData.standardize("test_table", 4)
    # PCAWeights.drop()
    # print(StdCVData.read("test_table", 0).collect())
    # print(StdCVData.read("test_table", 1).collect())
    # print(StdCVData.read("test_table", 2).collect())
    # print(StdCVData.read("test_table", 3).collect())
    # print(StdCVData.read("test_table", 4).collect())
    # PCAWeights.create()
    # StdCVData.calculate_pca_weights("test_table", 0)
    # StdCVData.calculate_pca_weights("test_table", 1)
    # StdCVData.calculate_pca_weights("test_table", 2)
    # StdCVData.calculate_pca_weights("test_table", 3)
    # StdCVData.calculate_pca_weights("test_table", 4)
    RenderData.create()
    # remove_table_name_from_cv_table("test_table")
    # print(get_cv_no_chunks("test_table"))
