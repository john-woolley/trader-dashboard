import sqlalchemy as sa
import psycopg2

CONN = "postgresql+psycopg2://trader_dashboard@0.0.0.0:5432/trader_dashboard"


def getconn():
    c = psycopg2.connect(user="trader_dashboard",
                         host="0.0.0.0", dbname="trader_dashboard")
    return c


def create_jobs_table():
    conn = sa.create_engine(CONN).connect()
    table = sa.Table(
        "jobs",
        sa.MetaData(),
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String),
        sa.Column("status", sa.String),
    )
    table.create(conn, checkfirst=True)
    conn.commit()
    conn.close()


def create_workers_table():
    conn = sa.create_engine(CONN).connect()
    table = sa.Table(
        "workers",
        sa.MetaData(),
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("job_id", sa.Integer, sa.ForeignKey("jobs.id")),
        sa.Column("name", sa.String),
        sa.Column("status", sa.String),
    )
    table.create(conn, checkfirst=True)
    conn.commit()
    conn.close()

def get_workers() -> str:
    conn = sa.create_engine(CONN).connect()
    worker_query = sa.select("*").select_from(sa.table("workers"))
    worker_list = list(map(lambda x: x[1], conn.execute(worker_query).fetchall()))
    workers = ','.join(worker_list)
    return workers