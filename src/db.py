"""
This module contains functions for interacting with a PostgreSQL database in 
the context of a trader dashboard application.
It provides functions for creating tables, inserting data,
retrieving data, and managing jobs and workers.
"""

import sqlalchemy as sa
import psycopg2
import pandas as pd
import cloudpickle

CONN = "postgresql+psycopg2://trader_dashboard@0.0.0.0:5432/trader_dashboard"


def getconn():
    """
    Returns a connection to the trader_dashboard database.
    """
    c = psycopg2.connect(
        user="trader_dashboard", host="0.0.0.0", dbname="trader_dashboard"
    )
    return c


def drop_jobs_table():
    """
    Drops the 'jobs' table from the database.
    """
    conn = sa.create_engine(CONN).connect()
    try:
        table = sa.Table("jobs", sa.MetaData(), autoload_with=conn)
        table.drop(conn, checkfirst=True)
        conn.commit()
    except sa.exc.DBAPIError as e:
        print(e)
    finally:
        conn.close()


def drop_workers_table():
    """
    Drops the 'workers' table from the database.
    """
    conn = sa.create_engine(CONN).connect()
    try:
        table = sa.Table("workers", sa.MetaData(), autoload_with=conn)
        table.drop(conn, checkfirst=True)
        conn.commit()
    except sa.exc.DBAPIError as e:
        print(e)
    finally:
        conn.close()


def create_jobs_table():
    """
    Creates a jobs table in the database.

    This function connects to the database, creates a metadata object,
    reflects the existing tables,
    defines the structure of the jobs table, and creates the table if it
    doesn't already exist.

    Returns:
        None
    """
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    table = sa.Table(
        "jobs",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String),
        sa.Column("status", sa.String),
    )
    table.create(conn, checkfirst=True)
    conn.commit()
    conn.close()


def create_workers_table():
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
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    jobs_table = sa.Table("jobs", metadata, autoload_with=conn)
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
    table.create(conn, checkfirst=True)
    conn.commit()
    conn.close()


def get_workers_by_name(job_name: str) -> str:
    """
    Retrieve the names of workers associated with a specific job name.

    Args:
        job_name (str): The name of the job.

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
            workers_table.join(jobs_table, jobs_table.c.id == workers_table.c.job_id)
        )
        .where(jobs_table.c.name == job_name)
    )
    res = conn.execute(worker_query).fetchall()
    worker_list = list(map(lambda x: x[0], res))
    workers = ",".join(worker_list)
    conn.close()
    return workers


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
            workers_table.join(jobs_table, jobs_table.c.id == workers_table.c.job_id)
        )
        .where(jobs_table.c.id == job_id)
    )
    res = conn.execute(worker_query).fetchall()
    worker_list = list(map(lambda x: x[0], res))
    workers = ",".join(worker_list)
    conn.close()
    return workers


def add_job(job_name: str) -> None:
    """
    Add a job to the database.

    Parameters:
    - job_name (str): The name of the job to add.

    Returns:
    - None
    """
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    jobs_table = sa.Table("jobs", metadata)
    ins = jobs_table.insert().values(name=job_name, status="idle")
    conn.execute(ins)
    conn.commit()
    conn.close()


def add_worker(worker_name: str, jobname: str) -> None:
    """
    Add a worker to the database with the specified worker name and job name.

    Args:
        worker_name (str): The name of the worker.
        jobname (str): The name of the job.

    Returns:
        None
    """

    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    job_table = sa.Table("jobs", metadata)
    job_query = sa.select(job_table.c.id).where(job_table.c.name == jobname)
    fetched = conn.execute(job_query).fetchone()
    assert fetched is not None, f"Job {jobname} not found"
    job_id = fetched[0]
    worker_table = sa.Table("workers", metadata)
    ins = worker_table.insert().values(name=worker_name, status="idle", job_id=job_id)
    conn.execute(ins)
    conn.commit()
    conn.close()


def create_cv_data_table():
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
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    table = sa.Table(
        "cv_data",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("table_name", sa.String),
        sa.Column("chunk", sa.Integer),
        sa.Column("data", sa.LargeBinary),
    )
    table.create(conn, checkfirst=True)
    conn.commit()
    conn.close()


def add_cv_data(table_name: str, data: bytes, i: int) -> None:
    """
    Add CV data to the specified table.

    Args:
        table_name (str): The name of the table to add the data to.
        data (bytes): The CV data to be added.
        i (int): The chunk index of the data.

    Returns:
        None
    """
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    table = sa.Table("cv_data", metadata)
    ins = table.insert().values(table_name=table_name, data=data, chunk=i)
    conn.execute(ins)
    conn.commit()
    conn.close()


def drop_cv_data_table():
    """
    Drops the 'cv_data' table from the database.

    This function connects to the database, retrieves the 'cv_data' table,
    and drops it.
    If the table does not exist, it does nothing.
    """
    conn = sa.create_engine(CONN).connect()
    try:
        table = sa.Table("cv_data", sa.MetaData(), autoload_with=conn)
        table.drop(conn, checkfirst=True)
        conn.commit()
    except sa.exc.DBAPIError as e:
        print(e)
    finally:
        conn.close()


def get_cv_data_by_name(table_name: str, i: int) -> bytes:
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
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    table = sa.Table("cv_data", metadata)
    query = (
        sa.select(table.c.data)
        .where(table.c.table_name == table_name)
        .where(table.c.chunk == i)
    )
    fetched = conn.execute(query).fetchone()
    assert fetched is not None, f"Data not found for {table_name} chunk {i}"
    res = fetched[0]
    conn.close()
    return res


def insert_raw_table(df: pd.DataFrame, table_name: str) -> None:
    """
    Inserts a pandas DataFrame into a SQL table.

    Args:
        df (pd.DataFrame): The DataFrame to be inserted.
        table_name (str): The name of the SQL table.

    Returns:
        None
    """
    conn = sa.create_engine(CONN).connect()
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()


def get_raw_table(table_name: str) -> pd.DataFrame:
    """
    Retrieves a raw table from the database.

    Args:
        table_name (str): The name of the table to retrieve.

    Returns:
        pd.DataFrame: The raw table data as a pandas DataFrame.
    """
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    df = pd.read_sql_table(table_name, conn)
    conn.close()
    return df


def chunk_df(df: pd.DataFrame, chunk_size: int) -> list:
    """
    Splits a DataFrame into chunks of a specified size.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        chunk_size (int): The size of each chunk.

    Returns:
        list: A list of DataFrame chunks.
    """
    chunks = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
    return chunks


def chunk_raw_table(table_name: str, chunk_size: int) -> list:
    """
    Chunk the raw table data into smaller chunks and insert them into the database table.

    Args:
        table_name (str): The name of the database table.
        chunk_size (int): The size of each chunk.

    Returns:
        list: The list of chunks.
    """
    df = get_raw_table(table_name)
    chunks = chunk_df(df, chunk_size)
    insert_chunked_table(chunks, table_name)
    return chunks


def insert_chunked_table(chunks: list, table_name: str) -> None:
    """
    Inserts a list of chunks into a table in the database.

    Args:
        chunks (list): A list of chunks to be inserted.
        table_name (str): The name of the table in the database.

    Returns:
        None
    """
    for i, chunk in enumerate(chunks):
        blob = cloudpickle.dumps(chunk)
        add_cv_data(table_name, blob, i)


def read_chunked_table(table_name: str, i: int) -> pd.DataFrame:
    """
    Read a chunked table from the database.

    Args:
        table_name (str): The name of the table.
        i (int): The index of the chunk.

    Returns:
        pd.DataFrame: The chunked table as a pandas DataFrame.
    """
    blob = get_cv_data_by_name(table_name, i)
    df = cloudpickle.loads(blob)
    return df


if __name__ == "__main__":
    drop_workers_table()
    drop_jobs_table()
    drop_cv_data_table()
    create_jobs_table()
    create_workers_table()
    create_cv_data_table()
    add_job("test")
    add_worker("worker1", "test")
    add_worker("worker2", "test")
    print(get_workers_by_name("test"))
    print(get_workers_by_id(1))
    test_df = pd.read_csv("trader-dashboard/data/master.csv")
    insert_raw_table(test_df, "test_table")
    print(get_raw_table("test_table"))
    chunk_raw_table("test_table", 1000)
    print(read_chunked_table("test_table", 0))
