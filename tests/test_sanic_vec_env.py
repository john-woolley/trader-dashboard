import pytest
from stable_baselines3.common.envs import DummyEnv
from sanic import Sanic
from src.sanic_vec_env import SanicVecEnv


@pytest.fixture
def vec_env():
    env_fns = [lambda: DummyEnv() for _ in range(3)]
    app = Sanic(__name__)
    return SanicVecEnv(env_fns, app, jobname="trader")


def test_reset(vec_env):
    obs = vec_env.reset()
    # Check if observations are returned for all environments
    assert len(obs) == 3


def test_step(vec_env):
    obs = vec_env.reset()
    actions = [vec_env.action_space.sample() for _ in range(3)]
    obs, rewards, dones, infos = vec_env.step(actions)
    # Check if observations are returned for all environments
    assert len(obs) == 3
    # Check if rewards are returned for all environments
    assert len(rewards) == 3
    assert len(dones) == 3  # Check if dones are returned for all environments
    assert len(infos) == 3  # Check if infos are returned for all environments


def test_close(vec_env):
    vec_env.close()
    assert vec_env.closed  # Check if the environment is closed


def test_get_images(vec_env):
    images = vec_env.get_images()
    # Check if images are returned for all environments
    assert len(images) == 3


def test_get_attr(vec_env):
    attr_values = vec_env.get_attr("attribute_name")
    # Check if attribute values are returned for all environments
    assert len(attr_values) == 3


def test_set_attr(vec_env):
    vec_env.set_attr("attribute_name", "attribute_value")
    attr_values = vec_env.get_attr("attribute_name")
    # Check if attribute values are set correctly
    assert attr_values == ["attribute_value"] * 3


def test_env_method(vec_env):
    method_results = vec_env.env_method("method_name", arg1=1, arg2=2)
    # Check if method results are returned for all environments
    assert len(method_results) == 3


def test_env_is_wrapped(vec_env):
    is_wrapped = vec_env.env_is_wrapped(DummyEnv)
    # Check if the environments are not wrapped
    assert is_wrapped == [False] * 3
