import os
from celery import Celery

os.environ["MKL_THREADING_LAYER"] = "GNU"
celery_app = Celery("trader-dashboard", include=["tasks"])
celery_app.conf.broker_url = "redis://redis:6379/0"
celery_app.conf.result_backend = "redis://redis:6379/0"
if __name__ == "__main__":
    celery_app.start(
        ("worker", "-E", "--loglevel=info", "--pool=threads", "--concurrency=1000")
    )
