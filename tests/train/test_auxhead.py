"""Reading-target action-width regressions."""

import jax.numpy as jnp

from train.auxhead import _discard_action_available


def test_discard_action_available_accepts_honor_and_red_discards():
    mask = jnp.zeros((88,), jnp.bool_)
    assert not bool(_discard_action_available(mask))
    for action in (30, 34, 35, 36):
        assert bool(_discard_action_available(mask.at[action].set(True)))
    assert bool(_discard_action_available(mask.at[71].set(True)))
    # Non-discard action ids must not enable discard-only reading targets.
    assert not bool(_discard_action_available(mask.at[37].set(True)))
    assert not bool(_discard_action_available(mask.at[87].set(True)))
