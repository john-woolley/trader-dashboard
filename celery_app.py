from celery import Celery

celery_app = Celery("trader-dashboard", include=["main"])
celery_app.conf.broker_url = 'redis://localhost:6379/0'

if __name__ == "__main__":
    celery_app.start(("-A", "main", "worker", "--loglevel=info"))