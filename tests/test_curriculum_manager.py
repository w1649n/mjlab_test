"""Tests for CurriculumManager."""

from unittest.mock import Mock

import pytest
import torch

from mjlab.managers.curriculum_manager import (
  CurriculumManager,
  CurriculumTermCfg,
  NullCurriculumManager,
)


# 2026-09-02 stair-training update: exercise class-term checkpoint state.
class _StatefulCurriculumTerm:
  def __init__(self, cfg: CurriculumTermCfg, env) -> None:
    del cfg, env
    self.progress = 0

  def __call__(self, env, env_ids):
    del env, env_ids
    self.progress += 1
    return torch.tensor(float(self.progress))

  def state_dict(self) -> dict[str, int]:
    return {"progress": self.progress}

  def load_state_dict(self, state_dict: dict[str, int]) -> None:
    self.progress = state_dict["progress"]


@pytest.fixture
def mock_env():
  env = Mock()
  env.num_envs = 2
  return env


def test_get_active_iterable_terms_handles_dict_and_scalar_state(mock_env):
  """Dict- and scalar-shaped curriculum states both yield flat value lists.

  Regression: the dict branch previously indexed `terms` (a list) by term
  name, raising TypeError. Only observable through callers that invoke
  get_active_iterable_terms, which no in-tree caller currently does.
  """

  def dict_state_func(env, env_ids):
    return {"a": torch.tensor(1.5), "b": 2.0}

  def scalar_state_func(env, env_ids):
    return torch.tensor(7.0)

  cfg = {
    "dict_term": CurriculumTermCfg(func=dict_state_func, params={}),
    "scalar_term": CurriculumTermCfg(func=scalar_state_func, params={}),
  }
  manager = CurriculumManager(cfg, mock_env)
  manager.compute()

  terms = dict(manager.get_active_iterable_terms(0))
  assert terms["dict_term"] == [1.5, 2.0]
  assert terms["scalar_term"] == [7.0]


# 2026-09-02 stair-training update: verify named class-term state round-trips.
def test_state_dict_round_trip_for_class_terms(mock_env):
  def stateless_term(env, env_ids):
    del env, env_ids
    return torch.tensor(0.0)

  manager = CurriculumManager(
    {
      "stateful": CurriculumTermCfg(func=_StatefulCurriculumTerm, params={}),
      "stateless": CurriculumTermCfg(func=stateless_term, params={}),
    },
    mock_env,
  )
  stateful_term = manager.get_term_cfg("stateful").func
  assert isinstance(stateful_term, _StatefulCurriculumTerm)

  manager.compute()
  manager.compute()
  saved_state = manager.state_dict()
  assert saved_state == {"stateful": {"progress": 2}}

  stateful_term.progress = 99
  manager.load_state_dict(
    {"stateful": {"progress": 2}, "removed_term": {"progress": 123}}
  )
  assert stateful_term.progress == 2

  # Missing state belongs to an older/different checkpoint and is a no-op.
  stateful_term.progress = 7
  manager.load_state_dict({"removed_term": {"progress": 123}})
  assert stateful_term.progress == 7


# 2026-09-02 stair-training update: inactive curricula expose the same no-op API.
def test_null_curriculum_manager_checkpoint_api():
  manager = NullCurriculumManager()
  assert manager.state_dict() == {}
  manager.load_state_dict({"unused": {"progress": 1}})
