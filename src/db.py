"""
This module contains functions for interacting with a PostgreSQL database in
the context of a trader dashboard application.
It provides functions for creating tables, inserting data,
retrieving data, and managing jobs and workers.
"""
import asyncio
import sqlalchemy as sa

# import psycopg2
import asyncpg
import polars as pl
import polars.selectors as cs
import cloudpickle

from sqlalchemy.sql import update
from sqlalchemy import MetaData, Table

from sqlalchemy.ext.asyncio import create_async_engine

import socket


CONN = f"postgresql+asyncpg://trader_dashboard:psltest@postgres:5432/trader_dashboard"
SYNC_CONN = (
    f"postgresql+psycopg2://trader_dashboard:psltest@postgres:5432/trader_dashboard"
)


class DBConnection:
    @staticmethod
    def getconn():
        """
        Returns a connection to the trader_dashboard database.
        """
        c = asyncpg.connect(
            user="trader_dashboard", host="127.0.0.1", dbname="trader_dashboard"
        )
        return c


class DBBase:
    pass


class RawData:
    @classmethod
    async def get_engine(cls):
        return create_async_engine(CONN)

    @classmethod
    async def get_metadata(cls) -> MetaData:
        """
        Get the metadata for a table in the database.

        Args:
            table_name (str): The name of the table.

        Returns:
            dict: The metadata for the table.
        """
        conn = await cls.get_engine()
        metadata = sa.MetaData()
        async with conn.begin() as conn:
            await conn.run_sync(metadata.reflect)
        return metadata

    @classmethod
    async def get_table(cls, table_name: str, metadata: MetaData) -> Table:
        """
        Get the metadata for a table in the database.

        Args:
            table_name (str): The name of the table.

        Returns:
            dict: The metadata for the table.
        """
        table = sa.Table(table_name, metadata)
        return table

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
            SYNC_CONN,
            if_table_exists="replace",
        )

    @classmethod
    async def get(cls, table_name: str) -> pl.LazyFrame:
        """
        Retrieves a raw table from the database.

        Args:
            table_name (str): The name of the table to retrieve.

        Returns:
            pd.DataFrame: The raw table data as a pandas DataFrame.
        """

        table = sa.Table(table_name, await cls.get_metadata())
        query = sa.select(table)
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            df = await conn.run_sync(lambda conn: pl.read_database(query, conn))
        return df.lazy()

    @classmethod
    async def chunk(
        cls, table_name: str, chunk_size: int = 1000, no_chunks: int = 0
    ) -> list:
        """
        Chunk the raw table data into smaller chunks and insert them into the
        database table.

        Args:
            table_name (str): The name of the database table.
            chunk_size (int): The size of each chunk.

        Returns:
            list: The list of chunks.
        """
        df = await cls.get(table_name)
        df = df.sort("date", "ticker")
        if not no_chunks:
            chunks = cls._chunk_df_by_size(df, chunk_size)
        else:
            chunks = cls._chunk_df_by_number(df, no_chunks)
        await cls.insert_chunked_table(chunks, table_name)
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
    async def insert_chunked_table(cls, chunks: list, table_name: str) -> None:
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
            await CVData.add(table_name, blob, i)


class Jobs:
    @classmethod
    async def get_engine(cls):
        return create_async_engine(CONN)

    @classmethod
    async def get_metadata(cls) -> MetaData:
        """
        Get the metadata for a table in the database.

        Args:
            table_name (str): The name of the table.

        Returns:
            dict: The metadata for the table.
        """
        metadata = sa.MetaData()
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            await conn.run_sync(metadata.reflect)
        return metadata

    @classmethod
    async def get_table(cls, metadata: MetaData) -> Table:
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            try:
                table = await conn.run_sync(
                    lambda conn: sa.Table("jobs", sa.MetaData(), autoload_with=conn)
                )

            except sa.exc.NoSuchTableError:
                table = sa.Table(
                    "jobs",
                    metadata,
                    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                    sa.Column("name", sa.String),
                    sa.Column("status", sa.String),
                )
        return table

    @classmethod
    async def create(cls):
        """
        Creates a jobs table in the database.

        This function connects to the database, creates a metadata object,
        reflects the existing tables,
        defines the structure of the jobs table, and creates the table if it
        doesn't already exist.

        Returns:
            None
        """
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            metadata = await cls.get_metadata()
            table = await cls.get_table(metadata)
            await conn.run_sync(table.create, checkfirst=True)
            await conn.commit()

    @classmethod
    async def add(cls, job_name: str) -> None:
        """
        Add a job to the database.

        Parameters:
        - job_name (str): The name of the job to add.

        Returns:
        - None
        """
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            metadata = await cls.get_metadata()
            table = await cls.get_table(metadata)
            ins = table.insert().values(name=job_name, status="pending")
            await conn.execute(ins)
            await conn.commit()

    @classmethod
    async def start(cls, job_name: str) -> None:
        """
        Start a job in the database.

        Args:
            job_name (str): The name of the job to start.

        Returns:
            None
        """
        conn = await cls.get_engine()
        metadata = await cls.get_metadata()
        table = await cls.get_table(metadata)
        query = (
            update(table)
            .where(table.c.name == job_name)
            .values(status="running")
        )
        async with conn.begin() as conn:
            await conn.execute(query)
            await conn.commit()

    @classmethod
    async def complete(cls, job_name: str) -> None:
        """
        Complete a job in the database.

        Args:
            job_name (str): The name of the job to complete.

        Returns:
            None
        """
        conn = await cls.get_engine()
        metadata = await cls.get_metadata()
        table = await cls.get_table(metadata)
        query = (
            update(table)
            .where(table.c.name == job_name)
            .values(status="complete")
        )
        async with conn.begin() as conn:
            await conn.execute(query)
            await conn.commit()

    @classmethod
    async def drop(cls):
        """
        Drops the 'jobs' table from the database.
        """
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            metadata = await cls.get_metadata()
            table = await cls.get_table(metadata)
            await conn.run_sync(table.drop, checkfirst=True)
            await conn.commit()

    @classmethod
    async def get(cls, jobname):
        """
        Get the status of a job.

        Args:
            jobname (str): The name of the job.

        Returns:
            str: The status of the job.
        """
        conn = await cls.get_engine()
        metadata = await cls.get_metadata()
        table = await cls.get_table(metadata)
        query = sa.select(table.c.status).where(table.c.name == jobname)
        async with conn.begin() as conn:
            fetch = await conn.execute(query)
            status = fetch.fetchone()
            return status[0]
    
    @classmethod
    async def get_all(cls):
        """
        Gets all jobs and statuses from the database.
        """
        conn = await cls.get_engine()
        metadata = await cls.get_metadata()
        table = await cls.get_table(metadata)
        query = sa.select(table.c.name, table.c.status)
        async with conn.begin() as conn:
            fetch = await conn.execute(query)
            res = fetch.fetchall()
            return res


class Workers:
    @classmethod
    async def get_engine(cls):
        return create_async_engine(CONN)

    @classmethod
    async def get_metadata(cls) -> MetaData:
        """
        Get the metadata for a table in the database.

        Args:
            table_name (str): The name of the table.

        Returns:
            dict: The metadata for the table.
        """
        metadata = sa.MetaData()
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            await conn.run_sync(metadata.reflect)
        return metadata

    @classmethod
    async def get_jobs_table(cls, metadata: MetaData) -> Table:
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            try:
                jobs_table = await conn.run_sync(
                    lambda conn: sa.Table("jobs", sa.MetaData(), autoload_with=conn)
                )
            except sa.exc.NoSuchTableError:
                jobs_table = sa.Table(
                    "jobs",
                    metadata,
                    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                    sa.Column("name", sa.String),
                    sa.Column("status", sa.String),
                )
            return jobs_table

    @classmethod
    async def get_table(cls, metadata: MetaData, jobs_table: Table) -> Table:
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            try:
                table = await conn.run_sync(
                    lambda conn: sa.Table("workers", sa.MetaData(), autoload_with=conn)
                )
            except sa.exc.NoSuchTableError:
                table = sa.Table(
                    "workers",
                    metadata,
                    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                    sa.Column(
                        "job_id",
                        sa.Integer,
                        sa.ForeignKey(jobs_table.c.id, ondelete="CASCADE"),
                    ),
                    sa.Column("name", sa.String),
                    sa.Column("status", sa.String),
                )
        return table

    @classmethod
    async def create(cls):
        """
        Creates a jobs table in the database.

        This function connects to the database, creates a metadata object,
        reflects the existing tables,
        defines the structure of the jobs table, and creates the table if it
        doesn't already exist.

        Returns:
            None
        """
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            metadata = await cls.get_metadata()
            jobs_table = await cls.get_jobs_table(metadata)
            table = await cls.get_table(metadata, jobs_table)
            await conn.run_sync(table.create, checkfirst=True)
            await conn.commit()

    @classmethod
    async def add(cls, worker_name: str, jobname: str) -> None:
        """
        Add a worker to the database with the specified worker name and job name.

        Args:
            worker_name (str): The name of the worker.
            jobname (str): The name of the job.

        Returns:
            None
        """
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            metadata = await cls.get_metadata()
            jobs_table = await cls.get_jobs_table(metadata)
            job_query = sa.select(jobs_table.c.id).where(jobs_table.c.name == jobname)
            fetch = await conn.execute(job_query)
            fetched = fetch.fetchone()
            assert fetched is not None, f"Job {jobname} not found"
            job_id = fetched[0]
            worker_table = await cls.get_table(metadata, jobs_table)
            ins = worker_table.insert().values(
                name=worker_name, status="idle", job_id=job_id
            )
            await conn.execute(ins)
            await conn.commit()

    @classmethod
    async def get_workers_by_name(cls, job_name: str) -> str:
        """
        Retrieve the names of workers associated with a specific job name.

        Args:
            job_name (str): The name of the job.

        Returns:
            str: A comma-separated string of worker names.

        """
        conn = await cls.get_engine()
        metadata = await cls.get_metadata()
        jobs_table = await cls.get_jobs_table(metadata)
        workers_table = await cls.get_table(metadata, jobs_table)

        worker_query = (
            sa.select(workers_table.c.name)
            .select_from(
                workers_table.join(
                    jobs_table, jobs_table.c.id == workers_table.c.job_id
                )
            )
            .where(jobs_table.c.name == job_name)
        )
        async with conn.begin() as conn:
            fetch = await conn.execute(worker_query)
            res = fetch.fetchall()
            worker_list = list(map(lambda x: x[0], res))
            workers = ",".join(worker_list)
            return workers

    @classmethod
    async def get_workers_by_id(cls, job_id: int) -> str:
        """
        Retrieve the names of workers associated with a given job ID.

        Args:
            job_id (int): The ID of the job.

        Returns:
            str: A comma-separated string of worker names.

        """
        conn = await cls.get_engine()
        metadata = await cls.get_metadata()
        jobs_table = await cls.get_jobs_table(metadata)
        workers_table = await cls.get_table(metadata, jobs_table)
        worker_query = (
            sa.select(workers_table.c.name)
            .select_from(
                workers_table.join(
                    jobs_table, jobs_table.c.id == workers_table.c.job_id
                )
            )
            .where(jobs_table.c.id == job_id)
        )
        async with conn.begin() as conn:
            fetch = await conn.execute(worker_query)
        res = fetch.fetchall()
        worker_list = list(map(lambda x: x[0], res))
        workers = ",".join(worker_list)
        return workers

    @classmethod
    async def drop(cls):
        """
        Drops the 'workers' table from the database.
        """

        conn = await cls.get_engine()
        async with conn.begin() as conn:
            metadata = await cls.get_metadata()
            jobs_table = await cls.get_jobs_table(metadata)
            table = await cls.get_table(metadata, jobs_table)
            await conn.run_sync(table.drop, checkfirst=True)
            await conn.commit()


class CVData:
    @classmethod
    async def get_engine(cls):
        return create_async_engine(CONN)

    @classmethod
    async def get_metadata(cls) -> MetaData:
        """
        Get the metadata for a table in the database.

        Args:
            table_name (str): The name of the table.

        Returns:
            dict: The metadata for the table.
        """
        metadata = sa.MetaData()
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            await conn.run_sync(metadata.reflect)
        return metadata

    @classmethod
    async def get_table(cls, metadata: MetaData) -> Table:
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            try:
                table = await conn.run_sync(
                    lambda conn: sa.Table("cv_data", sa.MetaData(), autoload_with=conn)
                )
            except sa.exc.NoSuchTableError:
                table = sa.Table(
                    "cv_data",
                    metadata,
                    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                    sa.Column("table_name", sa.String),
                    sa.Column("chunk", sa.Integer),
                    sa.Column("data", sa.LargeBinary),
                )
        return table

    @classmethod
    async def create(cls):
        """
        Creates a jobs table in the database.

        This function connects to the database, creates a metadata object,
        reflects the existing tables,
        defines the structure of the jobs table, and creates the table if it
        doesn't already exist.

        Returns:
            None
        """
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            metadata = await cls.get_metadata()
            table = await cls.get_table(metadata)
            await conn.run_sync(table.create, checkfirst=True)
            await conn.commit()

    @classmethod
    async def add(cls, table_name: str, data: bytes, i: int) -> None:
        """
        Add CV data to the specified table.

        Args:
            table_name (str): The name of the table to add the data to.
            data (bytes): The CV data to be added.
            i (int): The chunk index of the data.

        Returns:
            None
        """
        conn = await cls.get_engine()
        metadata = await cls.get_metadata()
        table = await cls.get_table(metadata)
        ins = table.insert().values(table_name=table_name, data=data, chunk=i)
        async with conn.begin() as conn:
            await conn.execute(ins)
            await conn.commit()

    @classmethod
    async def drop(cls):
        """
        Drops the 'jobs' table from the database.
        """
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            metadata = await cls.get_metadata()
            table = await cls.get_table(metadata)
            await conn.run_sync(table.drop, checkfirst=True)
            await conn.commit()

    @classmethod
    async def get(cls, table_name: str, i: int) -> bytes:
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
        conn = await cls.get_engine()
        metadata = await cls.get_metadata()
        table = await cls.get_table(metadata)
        query = (
            sa.select(table.c.data)
            .where(table.c.table_name == table_name)
            .where(table.c.chunk == i)
        )
        async with conn.begin() as conn:
            fetch = await conn.execute(query)
        fetched = fetch.fetchone()
        assert fetched is not None, f"Data not found for {table_name} chunk {i}"
        res = fetched[0]
        return res

    @classmethod
    async def get_table_names(cls) -> list:
        """
        Get the names of all tables in the cv_data table.

        Returns:
            list: A list of table names.
        """
        metadata = await cls.get_metadata()
        table = await cls.get_table(metadata)
        query = sa.select(table.c.table_name).distinct()
        fetch = await cls.conn.execute(query)
        res = fetch.fetchall()
        names = list(map(lambda x: x[0], res))
        return names

    @classmethod
    async def read(cls, table_name: str, i: int) -> pl.LazyFrame:
        """
        Read a chunked table from the database.

        Args:
            table_name (str): The name of the table.
            i (int): The index of the chunk.

        Returns:
            pd.DataFrame: The chunked table as a pandas DataFrame.
        """
        blob = await cls.get(table_name, i)
        df = cloudpickle.loads(blob)
        return df.lazy()

    @classmethod
    async def standardize(cls, table_name: str, chunk: int) -> None:
        """
        Standardize the columns of a chunked table in the database.

        Args:
            table_name (str): The name of the table.
            chunk (int): The index of the chunk.

        Returns:
            None
        """
        lazy = await cls.read(table_name, chunk)
        df = lazy.collect()
        if chunk == 0:
            std_lazy = await cls.read(table_name, 0)
            std_df = std_lazy.collect()
        else:
            dfs = asyncio.gather(*[cls.read(table_name, i) for i in range(chunk)])
            std_df = pl.concat([df.collect() for df in await dfs])
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
        await StdCVData.insert(table_name, chunk, blob)

    @classmethod
    async def get_cv_no_chunks(cls, table_name: str):
        """
        Get the number of chunks for a given table.

        Args:
            table_name (str): The name of the table.

        Returns:
            int: The number of chunks.
        """
        conn = await cls.get_engine()
        metdata = await cls.get_metadata()
        table = await cls.get_table(metdata)
        query = sa.select(table.c.chunk).where(table.c.table_name == table_name)
        async with conn.begin() as conn:
            fetch = await conn.execute(query)
        res = fetch.fetchall()
        return len(res)


class StdCVData:
    @classmethod
    async def get_engine(cls):
        return create_async_engine(CONN)

    @classmethod
    async def get_metadata(cls) -> MetaData:
        """
        Get the metadata for a table in the database.

        Args:
            table_name (str): The name of the table.

        Returns:
            dict: The metadata for the table.
        """
        metadata = sa.MetaData()
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            await conn.run_sync(metadata.reflect)
        return metadata

    @classmethod
    async def get_table(cls, metadata: MetaData) -> Table:
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            try:
                table = await conn.run_sync(
                    lambda conn: sa.Table(
                        "std_cv_data", sa.MetaData(), autoload_with=conn
                    )
                )
            except sa.exc.NoSuchTableError:
                table = sa.Table(
                    "std_cv_data",
                    metadata,
                    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                    sa.Column("table_name", sa.String),
                    sa.Column("chunk", sa.Integer),
                    sa.Column("data", sa.LargeBinary),
                )
        return table

    @classmethod
    async def create(cls):
        """

        Returns:
            None
        """
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            metadata = await cls.get_metadata()
            table = await cls.get_table(metadata)
            await conn.run_sync(table.create, checkfirst=True)
            await conn.commit()

    @classmethod
    async def insert(cls, table_name: str, chunk: int, data: bytes) -> None:
        """
        Insert standardized CV data into the database.

        Args:
            table_name (str): The name of the table.
            chunk (int): The index of the chunk.
            data (bytes): The standardized CV data.

        Returns:
            None
        """
        conn = await cls.get_engine()
        metadata = await cls.get_metadata()
        table = await cls.get_table(metadata)
        ins = table.insert().values(table_name=table_name, data=data, chunk=chunk)
        async with conn.begin() as conn:
            await conn.execute(ins)
            await conn.commit()

    @classmethod
    async def drop(cls):
        """
        Drops the 'workers' table from the database.
        """

        conn = await cls.get_engine()
        async with conn.begin() as conn:
            metadata = await cls.get_metadata()
            table = await cls.get_table(metadata)
            await conn.run_sync(table.drop, checkfirst=True)
            await conn.commit()

    @classmethod
    async def read(cls, table_name: str, chunk: int) -> pl.LazyFrame:
        """
        Read a standardized cv table from the database.

        Args:
            table_name (str): The name of the table.
            chunk (int): The index of the chunk.

        Returns:
            pd.DataFrame: The standardized cv table as a pandas DataFrame.
        """
        blob = await cls.get(table_name, chunk)
        df = cloudpickle.loads(blob)
        return df.lazy()

    @classmethod
    async def get(cls, table_name: str, chunk: int) -> bytes:
        """
        Retrieve standardized CV data by table name and chunk index.

        Args:
            table_name (str): The name of the table.
            chunk (int): The chunk index.

        Returns:
            bytes: The standardized CV data.
        """
        conn = await cls.get_engine()
        metadata = await cls.get_metadata()
        table = await cls.get_table(metadata)
        query = (
            sa.select(table.c.data)
            .where(table.c.table_name == table_name)
            .where(table.c.chunk == chunk)
        )
        async with conn.begin() as conn:
            fetch = await conn.execute(query)
        fetched = fetch.fetchone()
        assert fetched is not None, f"Data not found for {table_name} chunk {chunk}"
        res = fetched[0]
        return res


class RenderData:
    @classmethod
    async def get_engine(cls):
        return create_async_engine(CONN)

    @classmethod
    async def get_metadata(cls) -> MetaData:
        """
        Get the metadata for a table in the database.

        Args:
            table_name (str): The name of the table.

        Returns:
            dict: The metadata for the table.
        """
        metadata = sa.MetaData()
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            await conn.run_sync(metadata.reflect)
        return metadata

    @classmethod
    async def get_table(cls, metadata: MetaData) -> Table:
        conn = await cls.get_engine()
        async with conn.begin() as conn:
            try:
                table = await conn.run_sync(
                    lambda conn: sa.Table(
                        "render_data", sa.MetaData(), autoload_with=conn
                    )
                )
            except sa.exc.NoSuchTableError:
                table = sa.Table(
                    "render_data",
                    metadata,
                    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                    sa.Column("table_name", sa.String),
                    sa.Column("chunk", sa.Integer),
                    sa.Column("data", sa.LargeBinary),
                )
        return table

    @classmethod
    async def create(cls):
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
        conn = await cls.get_engine()
        metadata = await cls.get_metadata()
        table = await cls.get_table(metadata)
        async with conn.begin() as conn:
            await conn.run_sync(table.create, checkfirst=True)
            await conn.commit()

    @classmethod
    async def add(cls, table_name: str, data: pl.DataFrame, i: int) -> None:
        """
        Add render data to the specified table.

        Args:
            table_name (str): The name of the table to add the data to.
            data (bytes): The render data to be added.
            i (int): The chunk index of the data.

        Returns:
        None
        """
        conn = await cls.get_engine()
        metadata = await cls.get_metadata()
        table = await cls.get_table(metadata)
        blob = cloudpickle.dumps(data)
        ins = table.insert().values(table_name=table_name, data=blob, chunk=i)
        async with conn.begin() as conn:
            await conn.execute(ins)
            await conn.commit()

    @classmethod
    async def drop(cls):
        """
        Drops the 'workers' table from the database.
        """

        conn = await cls.get_engine()
        async with conn.begin() as conn:
            metadata = await cls.get_metadata()
            table = await cls.get_table(metadata)
            await conn.run_sync(table.drop, checkfirst=True)
            await conn.commit()

    @classmethod
    async def get(cls, table_name: str, i: int) -> bytes:
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
        conn = await cls.get_engine()
        metadata = await cls.get_metadata()
        table = await cls.get_table(metadata)
        query = (
            sa.select(table.c.data)
            .where(table.c.table_name == table_name)
            .where(table.c.chunk == i)
            .order_by(table.c.id)
        )
        async with conn.begin() as conn:
            fetch = await conn.execute(query)
        fetched = fetch.fetchall()
        assert fetched is not None, f"Data not found for {table_name} chunk {i}"
        res = fetched[-1].data
        return res

    @classmethod
    async def read(cls, table_name: str, i: int) -> pl.DataFrame:
        """
        Read a chunked table from the database.

        Args:
            table_name (str): The name of the table.
            i (int): The index of the chunk.

        Returns:
            pd.DataFrame: The chunked table as a pandas DataFrame.
        """
        blob = await cls.get(table_name, i)
        df = cloudpickle.loads(blob)
        return df


async def remove_table_name_from_cv_table(table_name: str) -> None:
    """
    Remove a table from the cv_data table.

    Args:
        table_name (str): The name of the table to remove.

    Returns:
        None
    """
    conn = create_async_engine(CONN)
    metadata = sa.MetaData()
    metadata.reflect(bind=conn.sync_engine)
    table = sa.Table("cv_data", metadata)
    query = sa.delete(table).where(table.c.table_name == table_name)
    await conn.execute(query)
    await conn.commit()


def get_workers():
    """
    Get all workers from the database.

    Returns:
        list: A list of workers.
    """
    conn = create_async_engine(CONN)
    metadata = sa.MetaData()
    metadata.reflect(bind=conn.sync_engine)
    workers_table = sa.Table("workers", metadata)
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
    conn = create_async_engine(CONN)
    metadata = sa.MetaData()
    metadata.reflect(bind=conn.sync_engine)
    jobs_table = sa.Table("jobs", metadata)
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
    conn = create_async_engine(CONN)
    metadata = sa.MetaData()
    metadata.reflect(bind=conn.sync_engine)
    jobs_table = sa.Table("jobs", metadata)
    workers_table = sa.Table("workers", metadata)
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


async def main():
    await Workers.drop()
    await Jobs.drop()
    await CVData.drop()
    await StdCVData.drop()
    await RenderData.drop()
    await Jobs.create()
    await Workers.create()
    await CVData.create()
    await Jobs.add("test")
    await Workers.add("worker1", "test")
    await Workers.add("worker2", "test")
    print(await Workers.get_workers_by_name("test"))
    print(await Workers.get_workers_by_id(1))
    test_df = (
        pl.read_csv("trader-dashboard/data/master.csv", try_parse_dates=True)
        .with_columns(pl.col("date").cast(pl.Date).alias("date"))
        .drop("")
        .sort("date", "ticker")
    )
    test_df = test_df.select(cs.all().forward_fill())
    RawData.insert(test_df, "test_table")
    print(await RawData.get("test_table"))
    await RawData.chunk("test_table", 63)
    print(await CVData.read("test_table", 0))
    await StdCVData.create()
    for i in range(await CVData.get_cv_no_chunks("test_table")):
        await CVData.standardize("test_table", i)
        print(await StdCVData.read("test_table", i))
    await RenderData.drop()
    await RenderData.create()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    # remove_table_name_from_cv_table("test_table")
    # print(get_cv_no_chunks("test_table"))
