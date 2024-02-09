import sqlalchemy as sa
import psycopg2
import datetime

#Create a pool
pool = sa.pool.QueuePool(
    lambda: psycopg2.connect(
        dbname="trader_dashboard",
        user="postgres",
        password="trader_dashboard",
        host="0.0.0.0",
        port=5432
    ), max_overflow=10, pool_size=5
    )

pool_engine = sa.create_engine("postgresql://postgres:trader_dashboard@0.0.0.0:5432/trader_dashboard", pool=pool)
pool_conn = pool_engine.connect()

#create table trades
metadata = sa.MetaData()
table = sa.Table(
    "trades",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("timestamp", sa.DateTime),
    sa.Column("symbol", sa.String(10)),
    sa.Column("price", sa.Float),
    sa.Column("quantity", sa.Float),
    sa.Column("side", sa.String(4)),
    sa.Column("order_type", sa.String(10)),
    sa.Column("exchange", sa.String(10)),
    sa.Column("status", sa.String(10)),
    sa.Column("user_id", sa.Integer)
    )
metadata.create_all(pool_conn)

# generate records
for i in range(10):
    ins = table.insert().values(
        timestamp=datetime.datetime.now(),
        symbol="AAPL",
        price=100.0,
        quantity=10.0,
        side="buy",
        order_type="market",
        exchange="NASDAQ",
        status="filled",
        user_id=1
    )
    pool_conn.execute(ins)