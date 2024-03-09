"""
This module contains functions for interacting with a PostgreSQL database in
the context of a trader dashboard application.
It provides functions for creating tables, inserting data,
retrieving data, and managing jobs and workers.
"""
import asyncio
import time
from typing import Optional

import sqlalchemy as sa

import polars as pl
import polars.selectors as cs
import cloudpickle

from sqlalchemy.sql import update
from sqlalchemy import MetaData, Table
from sqlalchemy.ext.asyncio import create_async_engine


CONN = f"postgresql+psycopg2://trader_dashboard:psltest@postgres:5432/trader_dashboard"
ASYNC_CONN = f"postgresql+asyncpg://trader_dashboard:psltest@postgres:5432/trader_dashboard"


class DBBase:
    @classmethod
    def get_engine(cls) -> sa.engine.Connection:
        return sa.create_engine(CONN, future=True).connect()
    
    @classmethod
    def get_async_engine(cls) -> sa.ext.asyncio.AsyncEngine:
        return create_async_engine(ASYNC_CONN)
        

    @classmethod
    def get_metadata(cls) -> MetaData:
        """
        Get the metadata for a table in the database.

        Args:
            table_name (str): The name of the table.

        Returns:
            MetaData: The metadata for the table.
        """
        conn = cls.get_engine()
        metadata = sa.MetaData()
        metadata.reflect(bind=conn)
        conn.close()
        return metadata
    
    @classmethod
    async def get_metadata_async(cls) -> MetaData:
        """
        Get the metadata for a table in the database.

        Args:
            table_name (str): The name of the table.

        Returns:
            MetaData: The metadata for the table.
        """
        conn = cls.get_async_engine()
        metadata = sa.MetaData()
        async with conn.connect() as c:
            await c.run_sync(metadata.reflect)
        return metadata

    @classmethod
    def get_table(cls, metadata: MetaData) -> Table:
        """
        Get the metadata for a table in the database.

        Args:
            table_name (str): The name of the table.

        Returns:
            dict: The metadata for the table.
        """
        return NotImplemented
    
    @classmethod
    async def get_table_async(cls, metadata: MetaData) -> Table:
        """
        Get the metadata for a table in the database.

        Args:
            table_name (str): The name of the table.

        Returns:
            dict: The metadata for the table.
        """
        return NotImplemented

    @classmethod
    def get(cls, table_name: str, jobname: Optional[str] = None, i: int = 0) -> pl.LazyFrame:
        """
        Retrieves a raw table from the database.

        Args:
            table_name (str): The name of the table to retrieve.

        Returns:
            pd.DataFrame: The raw table data as a pandas DataFrame.
        """

        table = sa.Table(table_name, cls.get_metadata())
        query = sa.select(table)
        conn = cls.get_engine()
        df = pl.read_database(query, conn)
        conn.close()
        return df.lazy()

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
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        table.create(checkfirst=True, bind=conn)
        conn.commit()
        conn.close()

    @classmethod
    def drop(cls):
        """
        Drops the 'jobs' table from the database.
        """
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        table.drop(checkfirst=True, bind=conn)
        conn.commit()
        conn.close()

    @classmethod
    def get_jobs_table(cls, metadata: MetaData) -> Table:
        conn = cls.get_engine()
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
        return jobs_table


class RawData(DBBase):
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
        df.write_database(table_name, CONN, if_table_exists="replace")

    @classmethod
    def chunk(cls, table_name: str, jobname: str, chunk_size: int = 1000, no_chunks: int = 0) -> list:
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
        if not no_chunks:
            chunks = cls._chunk_df_by_size(df, chunk_size)
        else:
            chunks = cls._chunk_df_by_number(df, no_chunks)
        cls.insert_chunked_table(chunks, table_name, jobname)
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
        dates = df.select("date").unique().sort(by="date").collect().to_series()
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
    def insert_chunked_table(cls, chunks: list, table_name: str, jobname:str) -> None:
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
            CVData.add(table_name, blob, i, jobname)


class Jobs(DBBase):
    @classmethod
    def get_table(cls, metadata: MetaData) -> Table:
        try:
            table = sa.Table("jobs", metadata, autoload_with=cls.get_engine())
        except sa.exc.NoSuchTableError:
            table = sa.Table(
                "jobs",
                metadata,
                sa.Column("name", sa.String, primary_key=True),
                sa.Column("status", sa.String),
                sa.Column("parent", sa.String)
            )
        return table

    @classmethod
    def add(cls, job_name: str, parent: Optional[str] = None) -> None:
        """
        Add a job to the database.

        Parameters:
        - job_name (str): The name of the job to add.

        Returns:
        - None
        """
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        ins = table.insert().values(name=job_name, parent=parent, status="pending")
        conn.execute(ins)
        conn.commit()
        conn.close()

    @classmethod
    def get_children(cls, job_name: str) -> list:
        """
        Get all the children of a job recursively.

        Args:
            job_name (str): The name of the job.

        Returns:
            list: A list of children.
        """
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        query = sa.select(table.c.name).where(table.c.parent == job_name)
        fetch = conn.execute(query)
        res = fetch.fetchall()
        children = list(map(lambda x: x[0], res))
        new_children = []
        for child in children:
            new_children.extend(cls.get_children(child))
        children.extend(new_children)
        conn.close()
        return children

    @classmethod
    def delete(cls, job_name: str) -> None:
        """
        Delete a job from the database.

        Parameters:
        - job_name (str): The name of the job to delete.

        Returns:
        - None
        """
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        children = cls.get_children(job_name)
        query = sa.delete(table).where(
            sa.or_(table.c.name == job_name, table.c.name.in_(children))
            )
        conn.execute(query)
        conn.commit()
        conn.close()

    @classmethod
    def start(cls, job_name: str) -> None:
        """
        Start a job in the database.

        Args:
            job_name (str): The name of the job to start.

        Returns:
            None
        """
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        query = update(table).where(table.c.name == job_name).values(status="running")
        conn.execute(query)
        conn.commit()
        conn.close()

    @classmethod
    def complete(cls, job_name: str) -> None:
        """
        Complete a job in the database.

        Args:
            job_name (str): The name of the job to complete.

        Returns:
            None
        """
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        query = update(table).where(table.c.name == job_name).values(status="complete")
        conn.execute(query)
        conn.commit()
        conn.close()

    @classmethod
    def get(cls, jobname):
        """
        Get the status of a job.

        Args:
            jobname (str): The name of the job.

        Returns:
            str: The status of the job.
        """
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        query = sa.select(table.c.status).where(table.c.name == jobname)
        fetch = conn.execute(query)
        conn.close()
        status = fetch.fetchone()
        return status[0]

    @classmethod
    def get_all(cls):
        """
        Gets all jobs and statuses from the database.
        """
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        query = sa.select(table.c.name, table.c.status, table.c.parent)
        fetch = conn.execute(query)
        conn.close()
        res = fetch.fetchall()
        return res


class Workers(DBBase):


    @classmethod
    def get_table(cls, metadata: MetaData) -> Table:
        conn = cls.get_engine()
        jobs_table = cls.get_jobs_table(metadata)
        try:
            table = sa.Table("workers", metadata, autoload_with=conn)
        except sa.exc.NoSuchTableError:
            table = sa.Table(
                "workers",
                metadata,
                sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                sa.Column(
                    "jobname",
                    sa.String,
                    sa.ForeignKey(jobs_table.c.name, ondelete="CASCADE"),
                ),
                sa.Column("name", sa.String),
                sa.Column("status", sa.String),
            )
        return table

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
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        jobs_table = cls.get_jobs_table(metadata)
        job_query = sa.select(jobs_table.c.name).where(jobs_table.c.name == jobname)
        fetch = conn.execute(job_query)
        fetched = fetch.fetchone()
        assert fetched is not None, f"Job {jobname} not found"
        job_id = fetched[0]
        worker_table = cls.get_table(metadata)
        ins = worker_table.insert().values(
            name=worker_name, status="idle", jobname=jobname
        )
        conn.execute(ins)
        conn.commit()

    @classmethod
    def get_workers_by_jobname(cls, job_name: str) -> str:
        """
        Retrieve the names of workers associated with a specific job name.

        Args:
            job_name (str): The name of the job.

        Returns:
            str: A comma-separated string of worker names.

        """
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        jobs_table = cls.get_jobs_table(metadata)
        workers_table = cls.get_table(metadata)

        worker_query = (
            sa.select(workers_table.c.name)
            .select_from(
                workers_table.join(
                    jobs_table, jobs_table.c.name == workers_table.c.jobname
                )
            )
            .where(jobs_table.c.name == job_name)
        )
        fetch = conn.execute(worker_query)
        res = fetch.fetchall()
        worker_list = list(map(lambda x: x[0], res))
        workers = ",".join(worker_list)
        return workers


class CVData(DBBase):
    @classmethod
    def get_table(cls, metadata: MetaData) -> Table:
        conn = cls.get_engine()
        try:
            table = sa.Table("cv_data", metadata, autoload_with=conn)
        except sa.exc.NoSuchTableError:
            jobs_table = cls.get_jobs_table(metadata)
            table = sa.Table(
                "cv_data",
                metadata,
                sa.Column("table_name", sa.String),
                sa.Column("chunk", sa.Integer, primary_key=True),
                sa.Column("data", sa.LargeBinary),
                sa.Column("jobname", sa.String, sa.ForeignKey(jobs_table.c.name, ondelete='CASCADE'), primary_key=True),
            )
        conn.close()
        return table
    
    @classmethod
    async def get_table_async(cls, metadata: MetaData) -> Table:
        conn = cls.get_async_engine()
        try:
            async with conn.begin() as c:
                table = await c.run_sync(lambda c: sa.Table("cv_data", metadata, autoload_with=c))
        except sa.exc.NoSuchTableError:
            jobs_table = cls.get_jobs_table(metadata)
            table = sa.Table(
                "cv_data",
                metadata,
                sa.Column("table_name", sa.String),
                sa.Column("chunk", sa.Integer, primary_key=True),
                sa.Column("data", sa.LargeBinary),
                sa.Column("jobname", sa.String, sa.ForeignKey(jobs_table.c.name, ondelete='CASCADE'), primary_key=True),
            )
        return table

    @classmethod
    def add(cls, table_name: str, data: bytes, i: int, jobname: str) -> None:
        """
        Add CV data to the specified table.

        Args:
            table_name (str): The name of the table to add the data to.
            data (bytes): The CV data to be added.
            i (int): The chunk index of the data.

        Returns:
            None
        """
        conn = cls.get_engine()
        sql = sa.text("""
            INSERT INTO cv_data (table_name, data, chunk, jobname)
            VALUES (:table_name, :data, :chunk, :jobname)
            ON CONFLICT (chunk, jobname)
            DO UPDATE SET data = excluded.data
        """)
        parms = {
            "table_name": table_name,
            "data": data,
            "chunk": i,
            "jobname": jobname,
        }
        conn.execute(sql, parms)
        conn.commit()
        conn.close()

    @classmethod
    def get(cls, table_name: str, jobname: Optional[str] = None, i: int = 0) -> pl.LazyFrame:
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
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        query = (
            sa.select(table.c.data)
            .where(table.c.table_name == table_name)
            .where(table.c.jobname == jobname)
            .where(table.c.chunk == i)
        )
        fetch = conn.execute(query)
        conn.close()
        fetched = fetch.fetchone()
        assert fetched is not None, f"Data not found for {table_name} chunk {i}"
        res = cloudpickle.loads(fetched[0])
        return res.lazy()
    
    @classmethod
    async def get_async(cls, table_name: str, jobname: Optional[str] = None, i: int = 0) -> pl.LazyFrame:
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
        conn = cls.get_async_engine()
        metadata = cls.get_metadata_async()
        table = cls.get_table(await metadata)
        query = (
            sa.select(table.c.data)
            .where(table.c.table_name == table_name)
            .where(table.c.jobname == jobname)
            .where(table.c.chunk == i)
        )
        async with conn.connect() as c:
            fetch = await c.execute(query)
        fetched = fetch.fetchone()
        assert fetched is not None, f"Data not found for {table_name} chunk {i}"
        res = cloudpickle.loads(fetched[0])
        return res.lazy()

    @classmethod
    def get_table_names(cls) -> list:
        """
        Get the names of all tables in the cv_data table.

        Returns:
            list: A list of table names.
        """
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        query = sa.select(table.c.table_name).distinct()
        fetch = conn.execute(query)
        conn.close()
        res = fetch.fetchall()
        names = list(map(lambda x: x[0], res))
        return names

    @classmethod
    def read(cls, table_name: str, jobname: str, i: int) -> pl.LazyFrame:
        """
        Read a chunked table from the database.

        Args:
            table_name (str): The name of the table.
            i (int): The index of the chunk.

        Returns:
            pd.DataFrame: The chunked table as a pandas DataFrame.
        """
        return cls.get(table_name, jobname, i)
    
    @classmethod
    async def read_async(cls, table_name: str, jobname: str, i: int) -> pl.LazyFrame:
        """
        Read a chunked table from the database.

        Args:
            table_name (str): The name of the table.
            i (int): The index of the chunk.

        Returns:
            pd.DataFrame: The chunked table as a pandas DataFrame.
        """
        return await cls.get_async(table_name, jobname, i)

    @classmethod
    def standardize(cls, table_name: str, jobname:str, chunk: int) -> None:
        """
        Standardize the columns of a chunked table in the database.

        Args:
            table_name (str): The name of the table.
            chunk (int): The index of the chunk.

        Returns:
            None
        """
        lazy = cls.read(table_name, jobname, chunk)
        df = lazy.collect()
        if chunk == 0:
            std_lazy = cls.read(table_name, jobname, 0)
            std_df = std_lazy.collect()
        else:
            dfs = [cls.read(table_name, jobname, i) for i in range(chunk)]
            std_df = pl.concat([df.collect() for df in dfs])
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
        df = df.sort("date")
        blob = cloudpickle.dumps(df)
        StdCVData.insert(table_name, chunk, blob, jobname)


    @classmethod
    async def standardize_async(cls, table_name: str, jobname:str, chunk: int) -> None:
        """
        Standardize the columns of a chunked table in the database.

        Args:
            table_name (str): The name of the table.
            chunk (int): The index of the chunk.

        Returns:
            None
        """
        lazy = await cls.read_async(table_name, jobname, chunk)
        df = lazy.collect()
        if chunk == 0:
            std_lazy = await cls.read_async(table_name, jobname, 0)
            std_df = std_lazy.collect()
        else:
            coros = [cls.read_async(table_name, jobname, i) for i in range(chunk)]
            dfs = await asyncio.gather(*coros)
            std_df = pl.concat([df.collect() for df in dfs])
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
        StdCVData.insert(table_name, chunk, blob, jobname)

    @classmethod
    def get_cv_no_chunks(cls, table_name: str, jobname: str) -> int:
        """
        Get the number of chunks for a given table.

        Args:
            table_name (str): The name of the table.

        Returns:
            int: The number of chunks.
        """
        conn = cls.get_engine()
        metdata = cls.get_metadata()
        table = cls.get_table(metdata)
        query = (
            sa.select(table.c.chunk)
            .where(table.c.table_name == table_name)
            .where(table.c.jobname == jobname)
        )
        fetch = conn.execute(query)
        conn.close()
        res = fetch.fetchall()
        return len(res)


class StdCVData(DBBase):
    @classmethod
    def get_table(cls, metadata: MetaData) -> Table:
        conn = cls.get_engine()
        try:
            table = sa.Table("std_cv_data", metadata, autoload_with=conn)
        except sa.exc.NoSuchTableError:
            jobs_table = cls.get_jobs_table(metadata)
            table = sa.Table(
                "std_cv_data",
                metadata,
                sa.Column("table_name", sa.String),
                sa.Column("chunk", sa.Integer, primary_key=True),
                sa.Column("data", sa.LargeBinary),
                sa.Column("jobname", sa.String, sa.ForeignKey(jobs_table.c.name, ondelete='CASCADE'), primary_key=True),
            )
        return table

    @classmethod
    def insert(cls, table_name: str, chunk: int, data: bytes, jobname: str) -> None:
        """
        Insert standardized CV data into the database.

        Args:
            table_name (str): The name of the table.
            chunk (int): The index of the chunk.
            data (bytes): The standardized CV data.

        Returns:
            None
        """
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        ins = table.insert().values(
            table_name=table_name, data=data, chunk=chunk, jobname=jobname
        )
        conn.execute(ins)
        conn.commit()
        conn.close()

    @classmethod
    async def insert_async(cls, table_name: str, chunk: int, data: bytes, jobname: str) -> None:
        """
        Insert standardized CV data into the database.

        Args:
            table_name (str): The name of the table.
            chunk (int): The index of the chunk.
            data (bytes): The standardized CV data.

        Returns:
            None
        """
        conn = cls.get_async_engine()
        metadata = await cls.get_metadata_async()
        table = await cls.get_table_async(metadata)
        ins = table.insert().values(
            table_name=table_name, data=data, chunk=chunk, jobname=jobname
        )
        async with conn.connect() as c:
            await c.execute(ins)

    @classmethod
    def read(cls, table_name: str, jobname: str, chunk: int) -> pl.LazyFrame:
        """
        Read a standardized cv table from the database.

        Args:
            table_name (str): The name of the table.
            chunk (int): The index of the chunk.

        Returns:
            pd.DataFrame: The standardized cv table as a pandas DataFrame.
        """
        return cls.get(table_name, jobname, chunk)

    @classmethod
    def get(cls, table_name: str, jobname:Optional[str] = None, i: int = 0) -> pl.LazyFrame:
        """
        Retrieve standardized CV data by table name and chunk index.

        Args:
            table_name (str): The name of the table.
            chunk (int): The chunk index.

        Returns:
            bytes: The standardized CV data.
        """
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        query = (
            sa.select(table.c.data)
            .where(table.c.table_name == table_name)
            .where(table.c.chunk == i)
            .where(table.c.jobname == jobname)
        )
        fetch = conn.execute(query)
        conn.close()
        fetched = fetch.fetchone()
        assert fetched is not None, f"Data not found for {table_name} chunk {i}"
        res = cloudpickle.loads(fetched[0])
        return res.lazy()


class RenderData(DBBase):
    @classmethod
    def get_table(cls, metadata: MetaData) -> Table:
        conn = cls.get_engine()
        try:
            table = sa.Table("render_data", metadata, autoload_with=conn)
        except sa.exc.NoSuchTableError:
            jobs_table = cls.get_jobs_table(metadata)
            table = sa.Table(
                "render_data",
                metadata,
                sa.Column("table_name", sa.String),
                sa.Column("chunk", sa.Integer, primary_key=True),
                sa.Column("data", sa.LargeBinary),
                sa.Column("jobname", sa.String, sa.ForeignKey(jobs_table.c.name, ondelete='CASCADE'), primary_key=True),
            )
        return table

    @classmethod
    def add(cls, table_name: str, data: pl.DataFrame, i: int, jobname: str) -> None:
        """
        Add render data to the specified table.

        Args:
            table_name (str): The name of the table to add the data to.
            data (bytes): The render data to be added.
            i (int): The chunk index of the data.

        Returns:
        None
        """
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        blob = cloudpickle.dumps(data)
        ins = table.insert().values(table_name=table_name, data=blob, chunk=i, jobname=jobname)
        conn.execute(ins)
        conn.commit()

    @classmethod
    def get(cls, table_name: str, jobname: Optional[str] = None, i: int = 0) -> pl.LazyFrame:
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
        conn = cls.get_engine()
        metadata = cls.get_metadata()
        table = cls.get_table(metadata)
        query = (
            sa.select(table.c.data)
            .where(table.c.table_name == table_name)
            .where(table.c.chunk == i)
            .where(table.c.jobname == jobname)
        )
        fetch = conn.execute(query)
        conn.close()
        fetched = fetch.fetchall()
        assert fetched is not None, f"Data not found for {table_name} chunk {i}"
        res = cloudpickle.loads(fetched[-1].data)
        return res.lazy()

    @classmethod
    def read(cls, table_name: str, jobname:str, i: int) -> pl.LazyFrame:
        """
        Read a chunked table from the database.

        Args:
            table_name (str): The name of the table.
            i (int): The index of the chunk.

        Returns:
            pd.DataFrame: The chunked table as a pandas DataFrame.
        """
        return cls.get(table_name, jobname, i)


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
    table = sa.Table("cv_data", metadata)
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
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
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
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
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

async def async_std():
    coros = [
        CVData.standardize_async("test_table", "test_job", i) 
        for i in range(CVData.get_cv_no_chunks("test_table", "test_job"))
    ]
    await asyncio.gather(*coros)

if __name__ == "__main__":
    CVData.drop()
    StdCVData.drop()
    RenderData.drop()
    Workers.drop()
    Jobs.drop()
    Jobs.create()
    Workers.create()
    CVData.create()
    StdCVData.create()
    StdCVData.create()
    RenderData.create()
