FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ADD requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
ADD celery_app.py /workspace/celery_app.py
ADD tasks.py /workspace/tasks.py
ADD ingestion.py /workspace/ingestion.py
ADD trader.py /workspace/trader.py
ADD cache.py /workspace/cache.py
ADD db.py /workspace/db.py
RUN export MKL_THREADING_LAYER=GNU
CMD [ "python", "/workspace/celery_app.py"]
RUN useradd -ms /bin/bash celery
RUN chown -R celery /workspace
USER celery