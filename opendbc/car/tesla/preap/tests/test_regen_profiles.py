"""Regen profiles (NAPRegenProfile) only ever add decel, stay inside the physical regen range,
and leave the relaxed profile identical to NAP's original feedforward."""
from types import SimpleNamespace

import numpy as np
import pytest

from opendbc.car.tesla.pedal.controller import get_zero_torque
from opendbc.car.tesla.preap import interface
from opendbc.car.tesla.preap.carcontroller import overspeed_pullback
from opendbc.car.tesla.preap.constants import OVERSPEED_MARGIN, REGEN_PROFILES, REGEN_PROFILE_DEFAULT
from opendbc.car.tesla.preap.nap_conf import REGEN_MAX
from opendbc.car.tesla.preap.virtual_das import VirtualDAS

SPEEDS = np.linspace(0.0, 40.0, 9)
DECELS = np.linspace(REGEN_MAX, 0.0, 31)


def _ff(regen_floor, a_cmd, v_ego):
  vdas = VirtualDAS()
  vdas.regen_floor = regen_floor
  return vdas._feedforward(a_cmd, v_ego)


def test_profile_floors_stay_inside_the_regen_range():
  for floor, gain in REGEN_PROFILES.values():
    assert REGEN_MAX <= floor < 0.0
    assert gain >= 0.0
  assert REGEN_PROFILES[0] == (REGEN_MAX, 0.0)  # relaxed = NAP unchanged


def test_relaxed_feedforward_is_nap_unchanged():
  vdas = VirtualDAS()
  for v in SPEEDS:
    for a in [*DECELS, 0.5, 1.5]:
      assert _ff(REGEN_MAX, a, v) == vdas.ff_model.get(a, v, get_zero_torque().get(v))


def test_profiles_are_monotonic_and_only_add_lift():
  floors = [REGEN_PROFILES[k][0] for k in (0, 1, 2)]  # relaxed, standard, aggressive
  for v in SPEEDS:
    for a in DECELS:
      di = [_ff(f, a, v) for f in floors]
      assert di[0] >= di[1] - 1e-9 >= di[2] - 2e-9, (v, a, di)


def test_positive_requests_are_untouched():
  for v in SPEEDS:
    for a in (0.0, 0.3, 1.0, 2.0):
      assert _ff(REGEN_PROFILES[2][0], a, v) == _ff(REGEN_MAX, a, v)


def test_full_lift_is_reached_at_the_profile_floor():
  for floor, _ in REGEN_PROFILES.values():
    for v in SPEEDS:
      assert _ff(floor, floor, v) == pytest.approx(_ff(REGEN_MAX, REGEN_MAX, v))
      assert _ff(floor, REGEN_MAX, v) == pytest.approx(_ff(REGEN_MAX, REGEN_MAX, v))  # never past it


def test_vdas_rejects_a_floor_outside_the_regen_range():
  for bad in (REGEN_MAX - 0.1, 0.0, 0.5):
    with pytest.raises(ValueError):
      VirtualDAS().update(-0.5, 20.0, 0.0, regen_floor=bad)


def test_overspeed_pullback_only_lowers_and_respects_the_floor():
  for floor, gain in REGEN_PROFILES.values():
    for v_set in (0.0, 10.0, 25.0):
      for v_ego in np.linspace(0.0, 40.0, 41):
        for req in (-1.5, -0.5, 0.0, 0.8):
          out = overspeed_pullback(req, v_ego, v_set, gain, floor)
          assert out <= req
          assert out >= min(req, floor)
          if v_set <= 0.0 or v_ego <= v_set + OVERSPEED_MARGIN:
            assert out == req


def test_overspeed_pullback_scales_with_overspeed():
  floor, gain = REGEN_PROFILES[1]
  assert overspeed_pullback(0.0, 20.0 + OVERSPEED_MARGIN + 1.0, 20.0, gain, floor) == pytest.approx(-gain)
  assert overspeed_pullback(0.0, 40.0, 20.0, gain, floor) == floor


@pytest.mark.parametrize("stored, expected", [
  (0, REGEN_PROFILES[0]), (1, REGEN_PROFILES[1]), (2, REGEN_PROFILES[2]),
  (7, REGEN_PROFILES[REGEN_PROFILE_DEFAULT]), ("junk", REGEN_PROFILES[REGEN_PROFILE_DEFAULT]),
  (None, REGEN_PROFILES[REGEN_PROFILE_DEFAULT]),
])
def test_regen_profile_param_falls_back_to_standard(monkeypatch, stored, expected):
  monkeypatch.setattr(interface, "_params", SimpleNamespace(get=lambda key, return_default=False: stored))
  assert interface.get_regen_profile() == expected
  assert interface.get_preap_accel_limits(20.0)[0] == expected[0]
