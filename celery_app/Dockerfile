FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ADD requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
ADD app.py /workspace/app.py
ADD tasks.py /workspace/tasks.py
ADD ingestion.py /workspace/ingestion.py
ADD trader.py /workspace/trader.py
ADD cache.py /workspace/cache.py
ADD train.py /workspace/train.py
ADD validate.py /workspace/validate.py
ADD db.py /workspace/db.py
ADD vec_env.py /workspace/vec_env.py
ADD memcache_lock.py /workspace/memcache_lock.py
ADD train.py /workspace/train.py
ADD validate.py /workspace/validate.py
ADD callbacks.py /workspace/callbacks.py
RUN export MKL_THREADING_LAYER=GNU
CMD [ "python", "/workspace/app.py"]
RUN useradd -ms /bin/bash celery
RUN chown -R celery /workspace
USER celery
