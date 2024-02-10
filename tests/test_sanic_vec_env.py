import pytest
from stable_baselines3.common.vec_env import DummyVecEnv
from sanic import Request
from src.sanic_vec_env import SanicVecEnv
from unittest.mock import Mock
import multiprocessing as mp
from src.db import add_job


# TODO: Fix this test
@pytest.fixture
def mock_request():
    request = Mock()
    request.app = Mock()
    server_info = Mock()
    manager = Mock()
    manager.durables = {"ident": "trader"}
    request.app.state.workers = 1
    request.app.listeners = {"main_process_ready": []}
    request.app.get_motd_data.return_value = ({"packages": ""}, {})
    request.app.state.server_info = [server_info]
    request.app.shared_ctx.mp_ctx = mp.get_context("spawn")
    request.app.manager = manager
    return request


@pytest.fixture
def vec_env(mock_request: Request):
    add_job("trader")
    env_fns = [lambda: DummyVecEnv() for _ in range(3)]
    return SanicVecEnv(env_fns, mock_request.app, jobname="trader")


# def test_reset(vec_env):
#     mp.set_start_method('spawn')  # or 'forkserver'
#     obs = vec_env.reset()
#     # Check if observations are returned for all environments
#     assert len(obs) == 3


# def test_step(vec_env):
#     obs = vec_env.reset()
#     actions = [vec_env.action_space.sample() for _ in range(3)]
#     obs, rewards, dones, infos = vec_env.step(actions)
#     # Check if observations are returned for all environments
#     assert len(obs) == 3
#     # Check if rewards are returned for all environments
#     assert len(rewards) == 3
#     assert len(dones) == 3  # Check if dones are returned for all environments
#     assert len(infos) == 3  # Check if infos are returned for all environments


# def test_close(vec_env):
#     vec_env.close()
#     assert vec_env.closed  # Check if the environment is closed


# def test_get_attr(vec_env):
#     attr_values = vec_env.get_attr("current_step")
#     # Check if attribute values are returned for all environments
#     assert len(attr_values) == 3


# def test_set_attr(vec_env):
#     vec_env.set_attr("attribute_name", "attribute_value")
#     attr_values = vec_env.get_attr("attribute_name")
#     # Check if attribute values are set correctly
#     assert attr_values == ["attribute_value"] * 3


# def test_env_method(vec_env):
#     method_results = vec_env.env_method("reset")
#     # Check if method results are returned for all environments
#     assert len(method_results) == 3


# def test_env_is_wrapped(vec_env):
#     is_wrapped = vec_env.env_is_wrapped(DummyVecEnv)
#     # Check if the environments are not wrapped
#     assert is_wrapped == [False] * 3
