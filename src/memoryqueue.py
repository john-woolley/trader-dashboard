
from multiprocessing import Process, Queue, Lock, Value

import torch

class Job:
    def __init__(self, job_id, memory_usage):
        self.job_id = job_id
        self.memory_usage = memory_usage

class MemoryQueue:
    def __init__(self, ctx):
        self.jobs_queue: Queue = ctx.Queue()
        self.gpu_capacity = torch.cuda.get_device_properties(0).total_memory
        self.lock = Lock()

    def add(self, job):
        with self.lock:
            if job.memory_usage <= self.gpu_capacity:
                self.jobs_queue.put(job)
            else:
                print(f"Job {job.job_id} exceeds available GPU memory. Not added to the queue.")

    def get(self):
        with self.lock:
            if not self.jobs_queue.empty():
                next_job = self.jobs_queue.get()
                return next_job
            else:
                return None