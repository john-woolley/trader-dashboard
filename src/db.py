import sqlalchemy as sa
import psycopg2

CONN = "postgresql+psycopg2://trader_dashboard@0.0.0.0:5432/trader_dashboard"


def getconn():
    c = psycopg2.connect(user="trader_dashboard",
                         host="0.0.0.0", dbname="trader_dashboard")
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
        sa.Column("job_id", sa.Integer, sa.ForeignKey("jobs.id", ondelete="CASCADE")),
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
        .select_from(workers_table.join(jobs_table, jobs_table.c.id == workers_table.c.job_id))
        .where(jobs_table.c.name == job_name)
    )
    res = conn.execute(worker_query).fetchall()
    worker_list = list(map(lambda x: x[0], res))
    workers = ','.join(worker_list)
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
        .select_from(workers_table.join(jobs_table, jobs_table.c.id == workers_table.c.job_id))
        .where(jobs_table.c.id == job_id)
    )
    res = conn.execute(worker_query).fetchall()
    worker_list = list(map(lambda x: x[0], res))
    workers = ','.join(worker_list)
    conn.close()
    return workers


def add_job(job_name: str) -> None:
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    jobs_table = sa.Table('jobs', metadata)
    ins = jobs_table.insert().values(name=job_name, status="idle")
    conn.execute(ins)
    conn.commit()
    conn.close()

def add_worker(worker_name: str, jobname: str) -> None:
    conn = sa.create_engine(CONN).connect()
    metadata = sa.MetaData()
    metadata.reflect(bind=conn)
    job_table = sa.Table('jobs', metadata)
    job_query = sa.select(job_table.c.id).where(job_table.c.name == jobname)
    job_id = conn.execute(job_query).fetchone()[0]
    worker_table = sa.Table('workers', metadata)
    ins = worker_table.insert().values(name=worker_name, status="idle", job_id=job_id)
    conn.execute(ins)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    drop_workers_table()
    drop_jobs_table()
    create_jobs_table()
    create_workers_table()
    add_job('test')
    add_worker('worker1', 'test')
    add_worker('worker2', 'test')
    print(get_workers_by_name('test'))
    print(get_workers_by_id(1))
