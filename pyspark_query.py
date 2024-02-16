from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("Python Spark SQL basic example")
    .config("spark.jars", "postgresql-42.7.1.jar")
    .getOrCreate()
)

df = (
    spark.read.format("jdbc")
    .option("url", "jdbc:postgresql://localhost:5432/trader_dashboard")
    .option("dbtable", "test_upload")
    .option("user", "trader_dashboard")
    .option("driver", "org.postgresql.Driver")
    .load()
)

df.printSchema()
