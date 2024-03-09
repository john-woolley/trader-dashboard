FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ADD requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
ADD sanic_app.py /workspace/sanic_app.py
ADD src /workspace/src
ADD db.py /workspace/db.py
RUN mkdir /workspace/data
CMD [ "python", "/workspace/sanic_app.py"]
EXPOSE 8004
RUN useradd -ms /bin/bash sanic
RUN chown -R sanic /workspace
USER sanic