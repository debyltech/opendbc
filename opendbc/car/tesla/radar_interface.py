from dataclasses import dataclass

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.tesla.values import DBC, CANBUS, CAR
from opendbc.car.interfaces import RadarInterfaceBase

# Optional NAP config import (available on device/runtime)
try:
  from opendbc.car.tesla.preap.nap_conf import nap_conf
except ImportError:
  nap_conf = None

try:
  from opendbc.car.tesla.preap.radar_table_freeze import BoschTableFreezeWatch
except ImportError:
  BoschTableFreezeWatch = None


BOSCH_POINT_BASE_ADDRESS = 0x310
BOSCH_POINT_ADDRESS_STRIDE = 3
BOSCH_TRACK_MAX_MISSED_CYCLES = 2
BOSCH_TRACK_MAX_DISTANCE_DELTA_M = 10.0
BOSCH_TRACK_MAX_VELOCITY_DELTA_MPS = 10.0

IGNORE_HW_FAIL_PARAM = "NAPRadarIgnoreHwFail"
IGNORE_HW_FAIL_PATH = "/data/params/d/NAPRadarIgnoreHwFail"


def _ignore_hw_fail_from_params():
  from openpilot.common.params import Params
  return bool(Params().get_bool(IGNORE_HW_FAIL_PARAM))


def _ignore_hw_fail_from_file(path=None):
  with open(path or IGNORE_HW_FAIL_PATH, "rb") as f:
    return f.read().strip() == b"1"


def _resolve_ignore_hw_fail():
  """Live NAPRadarIgnoreHwFail. Exceptions never force False if the file is 1."""
  try:
    if nap_conf is not None and bool(nap_conf.radar_ignore_hw_fail):
      return True
  except Exception:
    pass
  try:
    if _ignore_hw_fail_from_params():
      return True
  except Exception:
    pass
  try:
    if _ignore_hw_fail_from_file():
      return True
  except Exception:
    pass
  return False


@dataclass(frozen=True)
class BoschTrackObservation:
  d_rel: float
  y_rel: float
  v_rel: float
  a_rel: float
  yv_rel: float
  measured: bool


@dataclass
class BoschTrack:
  point: structs.RadarData.RadarPoint
  missed_cycles: int = 0
  # openpilot >= 0.11.2 deprecated RadarPoint.measured/aRel/yvRel and radard ignores them;
  # coasting tracks stay published (a lead must not vanish) and measured is kept here.
  measured: bool = True


class BoschTrackLifecycle:
  def __init__(self):
    self._slots: dict[int, BoschTrack] = {}
    self._next_track_id = 0

  @property
  def points(self):
    return [self._slots[slot].point for slot in sorted(self._slots)]

  def update(self, slot: int, observation: BoschTrackObservation | None):
    track = self._slots.get(slot)
    if observation is None:
      if track is None:
        return

      track.missed_cycles += 1
      if track.missed_cycles > BOSCH_TRACK_MAX_MISSED_CYCLES:
        del self._slots[slot]
      else:
        track.measured = False
      return

    if track is None or self._is_discontinuous(track.point, observation):
      point = structs.RadarData.RadarPoint()
      point.trackId = self._next_track_id
      self._next_track_id += 1
      track = BoschTrack(point)
      self._slots[slot] = track

    track.missed_cycles = 0
    track.point.dRel = observation.d_rel
    track.point.yRel = observation.y_rel
    track.point.vRel = observation.v_rel
    track.measured = observation.measured

  @staticmethod
  def _is_discontinuous(point, observation):
    distance_delta = abs(observation.d_rel - point.dRel)
    velocity_delta = abs(observation.v_rel - point.vRel)
    return (distance_delta > BOSCH_TRACK_MAX_DISTANCE_DELTA_M or
            velocity_delta > BOSCH_TRACK_MAX_VELOCITY_DELTA_MPS)


RADAR_START_ADDR = 0x410
RADAR_MSG_COUNT = 80  # 40 points * 2 messages each


def get_radar_can_parser(CP):
  if Bus.radar not in DBC[CP.carFingerprint]:
    return None

  messages = [('RadarStatus', 16)]
  for i in range(RADAR_MSG_COUNT // 2):
    messages.extend([
      (f'RadarPoint{i}_A', 16),
      (f'RadarPoint{i}_B', 16),
    ])

  return CANParser(DBC[CP.carFingerprint][Bus.radar], messages, 1)


class _ModelThreeYRadarInterface(RadarInterfaceBase):  # upstream radar for Model 3/Y
  def __init__(self, CP):
    super().__init__(CP)
    self.updated_messages = set()
    self.trigger_msg = RADAR_START_ADDR + RADAR_MSG_COUNT - 1

    self.radar_off_can = CP.radarUnavailable
    self.rcp = get_radar_can_parser(CP)

  def update(self, can_strings):
    if self.radar_off_can or self.rcp is None:
      return super().update(None)

    vls = self.rcp.update(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      return None

    rr = self._update(self.updated_messages)
    self.updated_messages.clear()

    return rr

  def _update(self, updated_messages):
    ret = structs.RadarData()
    if self.rcp is None:
      return ret

    if not self.rcp.can_valid:
      ret.errors.canError = True

    radar_status = self.rcp.vl['RadarStatus']
    if radar_status['shortTermUnavailable']:
      ret.errors.radarUnavailableTemporary = True
    if radar_status['sensorBlocked'] or radar_status['vehDynamicsError']:
      ret.errors.radarFault = True

    for i in range(RADAR_MSG_COUNT // 2):
      msg_a = self.rcp.vl[f'RadarPoint{i}_A']
      msg_b = self.rcp.vl[f'RadarPoint{i}_B']

      # Make sure msg A and B are together
      if msg_a['Index'] != msg_b['Index2']:
        continue

      if not msg_a['Tracked']:
        if i in self.pts:
          del self.pts[i]
        continue

      if i not in self.pts:
        self.pts[i] = structs.RadarData.RadarPoint()
        self.pts[i].trackId = self.track_id
        self.track_id += 1

      self.pts[i].dRel = msg_a['LongDist']
      self.pts[i].yRel = msg_a['LatDist']
      self.pts[i].vRel = msg_a['LongSpeed']

    ret.points = list(self.pts.values())
    return ret


class RadarInterface(RadarInterfaceBase):
  def __new__(cls, CP=None):
    # Legacy Bosch/Continental radars use this class; Model 3/Y keep upstream's parser.
    if CP is not None and CP.carFingerprint not in (CAR.TESLA_MODEL_S_HW1, CAR.TESLA_MODEL_X_HW1, CAR.TESLA_MODEL_S_HW2,
                                 CAR.TESLA_MODEL_S_PREAP, CAR.TESLA_MODEL_S_HW3):
      return _ModelThreeYRadarInterface(CP)
    return super().__new__(cls)

  def __init__(self, CP):
    super().__init__(CP)
    self.CP = CP

    self.continental_radar = CP.carFingerprint in (CAR.TESLA_MODEL_S_HW3, )
    self.bosch_radar = CP.carFingerprint in (CAR.TESLA_MODEL_S_HW1, CAR.TESLA_MODEL_X_HW1, CAR.TESLA_MODEL_S_HW2, CAR.TESLA_MODEL_S_PREAP)

    messages = []
    if self.continental_radar:
      messages.append(('RadarStatus', 16))
      self.num_points = 40
      self.trigger_msg = 1119
      self.radar_point_frq = 16
    elif self.bosch_radar:
      messages.append(('TeslaRadarSguInfo', 8))

      self.num_points = 32
      self.trigger_msg = 878
      self.radar_point_frq = 8

    if self.bosch_radar or self.continental_radar:
      for i in range(self.num_points):
        messages.extend([
          (f'RadarPoint{i}_A', self.radar_point_frq),
          (f'RadarPoint{i}_B', self.radar_point_frq),
        ])

    self.radar_off_can = CP.radarUnavailable
    if not CP.radarUnavailable:
      self.rcp = CANParser(DBC[CP.carFingerprint][Bus.radar], messages, CANBUS.radar)
    else:
      self.rcp = None
    print(f"[NAP] RadarInterface: radarUnavailable={CP.radarUnavailable}, radar_off_can={self.radar_off_can}, rcp={'active' if self.rcp else 'None'}")

    self.updated_messages = set()
    self.track_id = 0
    self.bosch_tracks = BoschTrackLifecycle()
    self.table_freeze = BoschTableFreezeWatch() if BoschTableFreezeWatch is not None else None
    # Offset is applied after any upside-down sign flip so it stays a
    # vehicle-frame translation (not Tinkla's -(LatDist + offset)).
    self.radar_offset = 0.0
    self.radar_direction = 1
    if self.CP.carFingerprint == CAR.TESLA_MODEL_S_PREAP and nap_conf is not None:
      self.radar_offset = float(nap_conf.radar_offset)
      if nap_conf.radar_upside_down:
        self.radar_direction = -1
    self.ignore_hw_fail = _resolve_ignore_hw_fail()

  def update(self, can_msgs):

    if self.radar_off_can or (self.rcp is None):
      return super().update(None)

    values = self.rcp.update(can_msgs)
    self.updated_messages.update(values)

    if self.trigger_msg not in self.updated_messages:
      return None

    ret = structs.RadarData()

    if self.rcp is None:
      return ret

    # Re-read every cycle: compiled params_pyx may not know the new key,
    # and the on-disk param is the source of truth when that throws.
    ignore_hw_fail = _resolve_ignore_hw_fail()
    self.ignore_hw_fail = ignore_hw_fail

    # Errors
    if not self.rcp.can_valid:
      ret.errors.canError = True

    ret.errors.radarFault = False
    ret.errors.radarUnavailableTemporary = False
    if self.continental_radar:
      radar_status = self.rcp.vl['RadarStatus']
      if radar_status['shortTermUnavailable']:
        ret.errors.radarUnavailableTemporary = True
      if radar_status['sensorBlocked'] or radar_status['vehDynamicsError']:
        ret.errors.radarFault = True
    elif self.bosch_radar:
      radar_status = self.rcp.vl['TeslaRadarSguInfo']
      # SGUFail is a summary lamp: calibration, VIN reject, dirty-behind-
      # nosecone, and worse. Tinkla ignored it on Pre-AP. IRITable's unit
      # raised it for the adjustment triad while tracks were still live.
      # Driving is how that calibration clears, so do not block engage.
      # HWFail is the dead-sensor bit and still blocks, unless
      # NAPRadarIgnoreHwFail is set (tracks can stay live on a false HWFail).
      # Frozen+SGUFail is stock-safe radarFault, but Ignore drops those
      # ghost tracks instead of bricking engage.
      if radar_status['RADC_HWFail'] and not ignore_hw_fail:
        ret.errors.radarFault = True
        if self.table_freeze is not None:
          self.table_freeze.reset()
      if radar_status['RADC_SensorDirty']:
        ret.errors.radarUnavailableTemporary = True

    # Radar tracks
    for i in range(self.num_points):
      msg_a = self.rcp.vl[f'RadarPoint{i}_A']
      msg_b = self.rcp.vl[f'RadarPoint{i}_B']

      if self.bosch_radar:
        point_a_address = BOSCH_POINT_BASE_ADDRESS + i * BOSCH_POINT_ADDRESS_STRIDE
        slot_addresses = {point_a_address, point_a_address + 1}
        observation = None
        if slot_addresses <= self.updated_messages:
          observation = self._parse_bosch_track(msg_a, msg_b)
        self.bosch_tracks.update(i, observation)
        continue

      # Make sure msg A and B are together
      if msg_a['Index'] != msg_b['Index2']:
        continue

      # Check if it's a valid track
      if not msg_a['Tracked']:
        if i in self.pts:
          del self.pts[i]
        continue

      # New track!
      if i not in self.pts:
        self.pts[i] = structs.RadarData.RadarPoint()
        self.pts[i].trackId = self.track_id
        self.track_id += 1

      # Parse track data
      self.pts[i].dRel = msg_a['LongDist']
      self.pts[i].yRel = msg_a['LatDist'] + self.radar_offset
      self.pts[i].vRel = msg_a['LongSpeed']

    ret.points = self.bosch_tracks.points if self.bosch_radar else list(self.pts.values())
    if self.bosch_radar and self.table_freeze is not None:
      frozen = self.table_freeze.update(ret.points)
      if frozen and bool(self.rcp.vl['TeslaRadarSguInfo']['RADC_SGUFail']):
        if ignore_hw_fail:
          # Stay engageable; never publish the frozen ghost table as leads.
          ret.points = []
        else:
          ret.errors.radarFault = True
    self.updated_messages.clear()
    return ret

  def _parse_bosch_track(self, msg_a, msg_b):
    if msg_a['Index'] != msg_b['Index2'] or not msg_a['Tracked']:
      return None

    if msg_a["LongDist"] > 250.0 or msg_a["LongDist"] <= 0 or msg_a["ProbExist"] < 50.0:
      return None

    return BoschTrackObservation(
      d_rel=msg_a['LongDist'],
      y_rel=self.radar_direction * msg_a['LatDist'] + self.radar_offset,
      v_rel=msg_a['LongSpeed'],
      a_rel=msg_a['LongAccel'],
      yv_rel=self.radar_direction * msg_b['LatSpeed'],
      measured=bool(msg_a['Meas']),
    )
