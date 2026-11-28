import pytest

from opendbc.car.tesla.preap import coop_steering
from opendbc.car.tesla.preap.coop_steering import BLEND_IN_TIME, DT, ENTRY_TIME, RELEASE_TIME, TORQUE_FULL, CoopSteering

SPEED = 15.0  # m/s


def engaged_coop(toggle=True):
  coop = CoopSteering()
  coop.update(0.0, 0.0, 0.0, SPEED, False, toggle)  # the toggle latches while steering is inactive
  return coop


def run(coop, frames, model=0.0, angle=0.0, torque=0.0, rate=0.0, speed=SPEED):
  out = model
  for _ in range(frames):
    out = coop.update(model, angle, torque, speed, True, True)
    angle += rate * DT
  return out, angle


def test_off_is_the_model_angle_whatever_the_torque():
  coop = engaged_coop(toggle=False)
  out, _ = run(coop, 100, model=3.0, angle=1.0, torque=1.8)
  assert out == 3.0


def test_toggle_waits_for_the_next_engage():
  coop = engaged_coop(toggle=False)
  for _ in range(10):
    assert coop.update(3.0, 1.0, 1.8, SPEED, True, True) == 3.0  # turned on mid-drive: still off
  coop.update(0.0, 0.0, 0.0, SPEED, False, True)
  run(coop, 20, model=3.0, angle=1.0, torque=1.8)
  assert coop.blend > 0.0


def test_hands_resisting_an_openpilot_turn_do_not_enter():
  coop = engaged_coop()
  out, _ = run(coop, 200, model=10.0, torque=1.8, rate=20.0)  # wheel turning 20 deg/s
  assert out == 10.0 and coop.blend == 0.0


def test_a_spike_shorter_than_the_entry_time_is_ignored():
  coop = engaged_coop()
  run(coop, int(ENTRY_TIME / DT) - 2, torque=1.8)
  out, _ = run(coop, 50, model=2.0, angle=2.0, torque=0.0)
  assert out == 2.0 and coop.blend == 0.0


def test_no_entry_below_the_minimum_speed():
  coop = engaged_coop()
  out, _ = run(coop, 50, model=2.0, angle=0.0, torque=1.8, speed=3.0)
  assert out == 2.0


def test_a_nudge_on_a_straight_follows_the_wheel_without_a_step():
  coop = engaged_coop()
  prev, _ = run(coop, 25)  # straight, hands resting
  out = prev
  for _ in range(60):  # 1.5 Nm push, wheel held where the driver puts it
    out = coop.update(0.0, 0.0, 1.5, SPEED, True, True)
    assert abs(out - prev) < 0.5  # no jump: the rack is already at the model angle
    prev = out
  assert coop.blend == pytest.approx((1.5 - coop_steering.TORQUE_ON) / (TORQUE_FULL - coop_steering.TORQUE_ON))
  # with a full push the target is the wheel plus a lead in the push direction
  for _ in range(int(BLEND_IN_TIME / DT) + 1):
    full = coop.update(0.0, 5.0, TORQUE_FULL, SPEED, True, True)
  assert full > 5.0
  assert coop.update(0.0, -5.0, -TORQUE_FULL, SPEED, True, True) < -5.0


def test_release_hands_back_to_the_model_within_the_release_time():
  coop = engaged_coop()
  run(coop, 60, torque=TORQUE_FULL)
  assert coop.blend == 1.0
  outs = [coop.update(4.0, 2.0, 0.0, SPEED, True, True) for _ in range(int(RELEASE_TIME / DT) + 1)]
  assert outs[-1] == pytest.approx(4.0)
  steps = [b - a for a, b in zip(outs, outs[1:], strict=False)]
  assert max(abs(s) for s in steps) <= 2.0 * DT / RELEASE_TIME + 1e-9  # eases back, no snap


def test_disengaging_clears_the_blend():
  coop = engaged_coop()
  run(coop, 60, torque=TORQUE_FULL)
  coop.update(0.0, 0.0, 0.0, SPEED, False, True)
  assert coop.blend == 0.0


def test_pushing_the_way_the_model_already_steers_never_holds_it_back():
  coop = engaged_coop()
  run(coop, 60, model=10.0, angle=5.0, torque=TORQUE_FULL)
  assert coop.blend == 1.0
  assert coop.update(10.0, 5.0, TORQUE_FULL, SPEED, True, True) == 10.0  # model is ahead in the push direction
  assert coop.update(10.0, 5.0, -TORQUE_FULL, SPEED, True, True) < 5.0  # pushing against it still follows the hands
