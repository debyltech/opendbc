"""Software co-op steering for pre-AP angle control.

The 2012 rack has no co-op mode (DAS_steeringControlType 2 is RESERVED), so a light push is read from the torsion-bar
torque instead: once the driver holds it from a nearly still wheel, the angle target follows the wheel plus a small lead
in the push direction, and hands back to the model over about a second after release. A hard push is unchanged: the rack
drops angle control itself (EPAS_eacErrorCode 3, observed at 3.2-5 Nm) and openpilot disengages. The result still goes
through apply_steer_angle_limits_vm, so the panda's angle limits are untouched.

Thresholds come from 80 min of logged EPAS frames: resting hands on a straight stay under 0.4-0.7 Nm (p99), but read
1.3-2 Nm (p95) while openpilot turns the wheel, so entry needs a still wheel. Replayed over 2.1 h of drives, 1.0 Nm for
0.1 s enters on every push that peaked at 1.5 Nm or more and on none under 1.2 Nm.
"""
import numpy as np

from opendbc.car.common.filter_simple import FirstOrderFilter

DT = 0.02  # s; updated with the 50 Hz steering command
TORQUE_ON = 1.0  # Nm held to enter; same as STEER_THRESHOLD, so only while openpilot already sees the wheel pressed
TORQUE_FULL = 2.0  # Nm for a full follow, well under the rack's ~3.2 Nm abort
TORQUE_OFF = 0.5  # Nm; below it the blend decays back to the model
ENTRY_TIME = 0.1  # s held above TORQUE_ON (EPAS torque updates at 25 Hz)
STILL_RATE = 5.0  # deg/s; entry only from a nearly still wheel
MIN_SPEED = 6.7  # m/s (15 mph)
LEAD_DEG_PER_NM = 0.5  # follow target lead beyond TORQUE_ON, in the push direction (light; tune after on-car logs)
RELEASE_TIME = 1.0  # s for a full blend to decay back to the model
TORQUE_TO_ANGLE_SIGN = 1.0  # positive steeringTorque turns steeringAngleDeg up: both are negated EPAS signals; logs mostly agree


class CoopSteering:
  def __init__(self):
    self.enabled = False
    self.blend = 0.0
    self._held = 0.0
    self._engaged = False
    self._angle_prev: float | None = None
    self._rate = FirstOrderFilter(0.0, 0.1, DT)

  def update(self, model_angle: float, measured_angle: float, driver_torque: float, v_ego: float,
             lat_active: bool, toggle: bool) -> float:
    rate = 0.0 if self._angle_prev is None else (measured_angle - self._angle_prev) / DT
    self._angle_prev = measured_angle
    self._rate.update(rate)

    if not lat_active:
      # the toggle is latched here, so flipping it mid-drive takes effect at the next engage
      self.enabled, self.blend, self._held, self._engaged = toggle, 0.0, 0.0, False
      return model_angle
    if not self.enabled:
      return model_angle

    torque = abs(driver_torque)
    self._held = self._held + DT if torque >= TORQUE_ON else 0.0
    if (not self._engaged and self._held >= ENTRY_TIME and abs(self._rate.x) < STILL_RATE
        and v_ego > MIN_SPEED):
      self._engaged = True

    if self._engaged and torque >= TORQUE_OFF:
      self.blend = max(self.blend, float(np.clip((torque - TORQUE_ON) / (TORQUE_FULL - TORQUE_ON), 0.0, 1.0)))
    else:
      self.blend = max(0.0, self.blend - DT / RELEASE_TIME)
      if self.blend == 0.0 and torque < TORQUE_OFF:
        self._engaged = False

    lead = TORQUE_TO_ANGLE_SIGN * np.sign(driver_torque) * LEAD_DEG_PER_NM * max(0.0, torque - TORQUE_ON)
    follow = measured_angle + lead
    return float((1.0 - self.blend) * model_angle + self.blend * follow)
