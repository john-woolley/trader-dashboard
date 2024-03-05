import logging
import asyncio
import multiprocessing as mp
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper
from typing import Any, Dict, Optional
from stable_baselines3.common.vec_env.patch_gym import _patch_env


from cache import cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    started_flag: mp.Value,
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[Dict[str, Any]] = {}
    started_flag.value = 1
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class CeleryVecEnv(SubprocVecEnv):
    def __init__(self, jobname, env_fns, start_method=None):
        self.waiting = False
        self.closed = False
        self.jobname = jobname
        n_envs = len(env_fns)
        self.n_envs = n_envs

        if start_method is None:
            start_method = "fork"
        self.ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[self.ctx.Pipe() for _ in range(n_envs)])
        self.workers = []
        i = 0
        coros = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            coro = self.start_worker(work_remote, remote, env_fn, i)
            coros.append(coro)
            i += 1

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(asyncio.gather(*coros))
        cache.set(f"{jobname}_started", 1)
        logger.debug(f"Started {jobname}")

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super(SubprocVecEnv, self).__init__(n_envs, observation_space, action_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.close()
            for i, worker in enumerate(self.workers):
                cache.delete(f"{self.jobname}.{i}.pid")
                worker.terminate()
                worker.join()
        self.closed = True

    async def start_worker(self, work_remote, remote, env_fn, i):
        logger.debug(f"Starting worker {i} for job {self.jobname}")
        
        started_flag = self.ctx.Value("i", 0)
        args = (work_remote, remote, CloudpickleWrapper(env_fn), started_flag)

        worker = self.ctx.Process(target=_worker, args=args, daemon=True)
        worker.start()
        self.workers.append(worker)
        
        while started_flag.value != 1:
            await asyncio.sleep(0.1)
        
        logger.info(f"Started worker {i} for job {self.jobname}")
        cache.set(f"{self.jobname}.{i}.pid", worker.pid)
        work_remote.close()
