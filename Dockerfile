FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ADD requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
ADD main.py /workspace/main.py
ADD src /workspace/src
RUN mkdir /workspace/data
CMD [ "python", "/workspace/main.py"]
EXPOSE 8004