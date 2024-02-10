import sqlalchemy as sa
import psycopg2
import pandas as pd
import cloudpickle

CONN = "postgresql+psycopg2://trader_dashboard@0.0.0.0:5432/trader_dashboard"


def getconn():
    c = psycopg2.connect(
        user="trader_dashboard", host="0.0.0.0", dbname="trader_dashboard"
    )
    return c


def drop_jobs_table():
    conn = sa.create_engine(CONN).connect()
    try:
        table = sa.Table("jobs", sa.MetaData(), autoload_with=conn)
        table.drop(conn, checkfirst=True)
        conn.commit()
    except Exception as e:
        print(e)
    finally:
        conn.close()


def drop_workers_table():
    conn = sa.create_engine(CONN).connect()
    try:
        table = sa.Table("workers", sa.MetaData(), autoload_with=conn)
        table.drop(conn, checkfirst=True)
        conn.commit()
    except Exception as e:
        print(e)
    finally:
        conn.close()


def create_jobs_table():
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
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    jobs_table = sa.Table("jobs", metadata, autoload_with=conn)
    table = sa.Table(
        "workers",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "job_id", sa.Integer, sa.ForeignKey(
                jobs_table.c.id, ondelete="CASCADE"
                )
        ),
        sa.Column("name", sa.String),
        sa.Column("status", sa.String),
    )
    table.create(conn, checkfirst=True)
    conn.commit()
    conn.close()


def get_workers_by_name(job_name: str) -> str:
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
        .where(jobs_table.c.name == job_name)
    )
    res = conn.execute(worker_query).fetchall()
    worker_list = list(map(lambda x: x[0], res))
    workers = ",".join(worker_list)
    conn.close()
    return workers


def get_workers_by_id(job_id: int) -> str:
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
    conn.close()
    return workers


def add_job(job_name: str) -> None:
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    jobs_table = sa.Table("jobs", metadata)
    ins = jobs_table.insert().values(name=job_name, status="idle")
    conn.execute(ins)
    conn.commit()
    conn.close()


def add_worker(worker_name: str, jobname: str) -> None:
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    job_table = sa.Table("jobs", metadata)
    job_query = sa.select(job_table.c.id).where(job_table.c.name == jobname)
    fetched = conn.execute(job_query).fetchone()
    assert fetched is not None, f"Job {jobname} not found"
    job_id = fetched[0]
    worker_table = sa.Table("workers", metadata)
    ins = worker_table.insert().values(
        name=worker_name, status="idle", job_id=job_id
        )
    conn.execute(ins)
    conn.commit()
    conn.close()


def create_cv_data_table():
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
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    table = sa.Table("cv_data", metadata)
    ins = table.insert().values(table_name=table_name, data=data, chunk=i)
    conn.execute(ins)
    conn.commit()
    conn.close()


def drop_cv_data_table():
    conn = sa.create_engine(CONN).connect()
    try:
        table = sa.Table("cv_data", sa.MetaData(), autoload_with=conn)
        table.drop(conn, checkfirst=True)
        conn.commit()
    except Exception as e:
        print(e)
    finally:
        conn.close()


def get_cv_data_by_name(table_name: str, i: int) -> bytes:
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
    conn = sa.create_engine(CONN).connect()
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()


def get_raw_table(table_name: str) -> pd.DataFrame:
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    df = pd.read_sql_table(table_name, conn)
    conn.close()
    return df


def chunk_df(df: pd.DataFrame, chunk_size: int) -> list:
    chunks = [df[i: i + chunk_size] for i in range(0, len(df), chunk_size)]
    return chunks


def chunk_raw_table(table_name: str, chunk_size: int) -> list:
    df = get_raw_table(table_name)
    chunks = chunk_df(df, chunk_size)
    insert_chunked_table(chunks, table_name)
    return chunks


def insert_chunked_table(chunks: list, table_name: str) -> None:
    for i, chunk in enumerate(chunks):
        blob = cloudpickle.dumps(chunk)
        add_cv_data(table_name, blob, i)


def read_chunked_table(table_name: str, i: int) -> pd.DataFrame:
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
