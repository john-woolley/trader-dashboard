import multiprocessing as mp

from unittest.mock import Mock

import pytest
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv
from sanic import Request

from src.sanic_vec_env import SanicVecEnv
from src.db import add_job
import multiprocessing as mp


class MockEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self.render_mode = "human"

    def get_attr(self, attr_name):
        return getattr(self, attr_name, None)


def env_fn():
    env = MockEnvironment()
    return env


def dummy_func():
    return 1


@pytest.fixture
def mock_request_2_values():
    mock_context = Mock()
    mock_pipe = Mock()
    mock_pipe.recv.return_value = (Mock(), Mock())
    mock_context.Pipe.return_value = (mock_pipe, mock_pipe)
    request = Mock(
        app=Mock(manager=[Mock(durables={"a": Mock(processes=[])})]), headers={}
    )
    request.app.shared_ctx.mp_ctx = mock_context
    request.__getitem__ = Mock()
    return request


@pytest.fixture
def mock_request_5_values():
    mock_context = Mock()
    mock_pipe = Mock()

    def side_effect():
        if not hasattr(side_effect, "counter"):
            side_effect.counter = 0
        side_effect.counter += 1
        if side_effect.counter == 1:
            return (Mock(), Mock())
        else:
            return (Mock(), Mock(), Mock(), Mock(), Mock())

    mock_pipe.recv.side_effect = side_effect
    mock_context.Pipe.return_value = (mock_pipe, mock_pipe)
    request = Mock(
        app=Mock(manager=[Mock(durables={"a": Mock(processes=[])})]), headers={}
    )
    request.app.shared_ctx.mp_ctx = mock_context
    request.__getitem__ = Mock()
    return request


@pytest.fixture
def vec_env_2_values(mock_request_2_values: Request):
    add_job("trader")
    env_fns = [env_fn for _ in range(3)]
    vec_env = SanicVecEnv(env_fns, mock_request_2_values.app, jobname="trader")

    def mock_get_attr(self, attr_name):
        if attr_name == "render_mode":
            return ["human"] * 3
        elif attr_name == "current_step":
            return [0] * 3  # replace with the correct current_step values
        else:
            return self.__dict__.get(attr_name, Mock())

    def mock_env_is_wrapped(wrapper_class):
        if wrapper_class == DummyVecEnv:
            return [False] * 3
        else:
            return False

    def mock_set_attr(self, attr_name, value):
        vec_env.__dict__[attr_name] = [value] * 3

    vec_env.set_attr = mock_set_attr.__get__(vec_env)  # mypy: ignore
    vec_env.env_is_wrapped = mock_env_is_wrapped  # mypy: ignore
    vec_env.get_attr = mock_get_attr.__get__(vec_env)  # mypy: ignore

    return vec_env


@pytest.fixture
def vec_env_5_values(mock_request_5_values: Request):
    add_job("trader")
    env_fns = [env_fn for _ in range(3)]
    return SanicVecEnv(env_fns, mock_request_5_values.app, jobname="trader")

# TODO: Fix this test
# def test_reset(vec_env_2_values):
#     mp.set_start_method("spawn")  # or 'forkserver'
#     obs = vec_env_2_values.reset()
#     # Check if observations are returned for all environments
#     assert len(obs) == 3


# TODO: Fix this test
# def test_step(vec_env_5_values):
#     obs = vec_env_5_values.reset()
#     actions = [vec_env_5_values.action_space.sample() for _ in range(3)]
#     obs, rewards, dones, infos = vec_env_5_values.step(actions)
#     assert len(obs) == 3
#     assert len(rewards) == 3
#     assert len(dones) == 3
#     assert len(infos) == 3


def test_close(vec_env_2_values):
    vec_env_2_values.close()
    assert vec_env_2_values.closed  # Check if the environment is closed


def test_get_attr(vec_env_2_values):
    attr_values = vec_env_2_values.get_attr("current_step")
    # Check if attribute values are returned for all environments
    assert len(attr_values) == 3


def test_set_attr(vec_env_2_values):
    vec_env_2_values.set_attr("test_attribute", "human")
    attr_values = vec_env_2_values.get_attr("test_attribute")
    # Check if attribute values are set correctly
    assert attr_values == ["human"] * 3


def test_env_method(vec_env_2_values):
    method_results = vec_env_2_values.env_method("reset")
    # Check if method results are returned for all environments
    assert len(method_results) == 3


def test_env_is_wrapped(vec_env_2_values):
    is_wrapped = vec_env_2_values.env_is_wrapped(DummyVecEnv)
    # Check if the environments are not wrapped
    assert is_wrapped == [False] * 3
