
from multiprocessing import Queue, Lock, Value

import torch

class Job:
    def __init__(self, job_id, memory_usage, fn_name, args, cv_periods):
        self.job_id = job_id
        self.memory_usage = memory_usage
        self.fn_name = fn_name
        self.args = args
        self.cv_periods = cv_periods

class MemoryQueue:
    def __init__(self, ctx):
        self.jobs_queue = ctx.Queue()
        self.lock = Lock()
        self.estimated_memory_usage = Value("i", 0)

    @property
    def gpu_capacity(self):
        return torch.cuda.get_device_properties(0).total_memory

    def add(self, job):
        with self.lock:
            if job.memory_usage <= self.gpu_capacity:
                self.jobs_queue.put(job)
                self.estimated_memory_usage.value += job.memory_usage
            else:
                print(f"Job {job.job_id} exceeds available GPU memory. Not added to the queue.")

    def get(self):
        with self.lock:
            if not self.jobs_queue.empty():
                next_job = self.jobs_queue.get()
                self.estimated_memory_usage.value -= next_job.memory_usage
                return next_job
            else:
                return None