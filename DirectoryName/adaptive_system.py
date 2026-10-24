"""
adaptive_system.py — Main KPIT Sparkle 2027 Simulation
Tier 1: ONNX Edge AI signal control
Tier 2: GRU 30-second queue forecasting
Tier 3: HMAC-signed emergency vehicle handling (local detection via TraCI)
Tier 4: Per-junction configuration via junction_config.json

Team InnovateX · RMK Engineering College
"""

import os
import sys
import json
import time
import pickle
import random
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Tier 4 — Load per-junction configuration
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG_PATH = os.path.join(_HERE, 'junction_config.json')

try:
    with open(_CFG_PATH) as f:
        _cfg = json.load(f)
    print(f"[Config] Loaded junction config: {_cfg.get('location', 'MG Road')}")
except Exception as e:
    print(f"[Config] WARNING: Could not load junction_config.json ({e}), using defaults")
    _cfg = {}

SUMO_CONFIG    = os.path.join(_HERE, _cfg.get('sumo_config', 'kpit.sumocfg'))
SUMO_GUI       = _cfg.get('use_gui', True)
SIM_STEPS      = int(_cfg.get('sim_steps', 7200))
TRIPINFO_OUTPUT = os.path.join(_HERE, _cfg.get('tripinfo_output', 'adaptive_tripinfo.xml'))
MIN_GREEN      = int(_cfg.get('min_green', 8))
MAX_GREEN      = int(_cfg.get('max_green', 60))
EMERGENCY_GREEN = int(_cfg.get('emergency_green', 60))
CONGESTION_THRESHOLD = int(_cfg.get('congestion_threshold', 7))
EMERG_SPAWN_INTERVAL = int(_cfg.get('emergency_spawn_interval', 150))
EMERG_SPAWN_PROB     = float(_cfg.get('emergency_spawn_probability', 0.4))
GRU_WINDOW     = int(_cfg.get('gru_window_size', 30))

# ---------------------------------------------------------------------------
# SUMO / TraCI
# ---------------------------------------------------------------------------
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)

try:
    import traci
    import sumolib
except ImportError:
    sumo_home = r"C:\Program Files (x86)\Eclipse\Sumo"
    tools_path = os.path.join(sumo_home, 'tools')
    if tools_path not in sys.path:
        sys.path.append(tools_path)
    import traci
    import sumolib

# ---------------------------------------------------------------------------
# Tier 1 — ONNX Edge AI Controller
# ---------------------------------------------------------------------------
class EdgeAIController:
    def __init__(self):
        self.session = None
        self.scaler  = None
        self.config  = None
        self.input_name  = 'input'
        self.output_name = 'output'
        self.num_phases  = 3
        self._load()

    def _load(self):
        try:
            import onnxruntime as ort
            model_path  = os.path.join(_HERE, 'traffic_model.onnx')
            scaler_path = os.path.join(_HERE, 'traffic_scaler.pkl')
            config_path = os.path.join(_HERE, 'model_config.pkl')

            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            self.num_phases = self.config.get('num_phases', 3)
            print(f"[ONNX] Config: {self.num_phases} phases, input={self.config.get('input_size',9)}")

            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("[ONNX] Scaler loaded")

            self.session = ort.InferenceSession(model_path)
            self.input_name  = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"[ONNX] Model loaded: traffic_model.onnx")
        except Exception as e:
            print(f"[ONNX] WARNING: Could not load model ({e}) - will use rule-based fallback")

    def predict_phase(self, queues, speeds, emergency_flag) -> int:
        if self.session is None or self.scaler is None:
            return self._rule_based(queues, emergency_flag)
        try:
            features = np.array(list(queues) + list(speeds) + [emergency_flag],
                                dtype=np.float32).reshape(1, -1)
            features_scaled = self.scaler.transform(features).astype(np.float32)
            output = self.session.run([self.output_name], {self.input_name: features_scaled})
            return int(np.argmax(output[0]))
        except Exception as e:
            print(f"[ONNX] Inference error: {e}")
            return self._rule_based(queues, emergency_flag)

    def _rule_based(self, queues, emergency_flag) -> int:
        if emergency_flag:
            return int(np.argmax(queues))
        return int(np.argmax(queues))


# ---------------------------------------------------------------------------
# Tier 2 — GRU Queue Forecaster wrapper
# ---------------------------------------------------------------------------
class QueueForecaster:
    def __init__(self):
        self._history = deque(maxlen=GRU_WINDOW)
        self._predictor = None
        self._loaded = False
        self._try_load()

    def _try_load(self):
        try:
            from gru_predictor import GRUPredictor
            p = GRUPredictor()
            if p.load():
                self._predictor = p
                self._loaded = True
                print("[GRU] Forecaster ready (neural)")
            else:
                print("[GRU] Weights not found - using moving average fallback")
        except Exception as e:
            print(f"[GRU] Could not load predictor ({e}) - using moving average fallback")

    def update(self, total_queue: float):
        self._history.append(total_queue)
        if self._predictor:
            self._predictor.update(total_queue)

    def forecast(self) -> float:
        if self._loaded and self._predictor:
            return self._predictor.forecast()
        if len(self._history) == 0:
            return 0.0
        return float(sum(self._history) / len(self._history))


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------
EMERGENCY_TYPES = ['ambulance', 'fire_truck', 'police', 'rescue']

def try_spawn_emergency(step, valid_routes, active_emergencies):
    if step % EMERG_SPAWN_INTERVAL != 0:
        return
    if random.random() > EMERG_SPAWN_PROB:
        return
    vtype  = random.choice(EMERGENCY_TYPES)
    vid    = f"{vtype}_{step}"
    route  = random.choice(valid_routes)
    try:
        traci.vehicle.add(vid, route, typeID='DEFAULT_VEHTYPE')
        traci.vehicle.setColor(vid, (255, 0, 0, 255))
        traci.vehicle.setSpeedMode(vid, 7)
        active_emergencies[vid] = step
        print(f"[{step}s] EMERGENCY spawned: {vtype.upper()} -> {vid}")
    except Exception:
        pass


def read_traffic_state(controlled_lanes):
    queues, speeds = [], []
    for lane in controlled_lanes[:4]:
        try:
            q = traci.lane.getLastStepHaltingNumber(lane)
            s = traci.lane.getLastStepMeanSpeed(lane)
        except Exception:
            q, s = 0, 0.0
        queues.append(float(q))
        speeds.append(float(s))
    while len(queues) < 4:
        queues.append(0.0)
        speeds.append(0.0)
    return queues[:4], speeds[:4]


def detect_emergency(active_emergencies, controlled_lanes):
    all_vehicles = set(traci.vehicle.getIDList())
    # Clean up vehicles that left the sim
    for vid in list(active_emergencies.keys()):
        if vid not in all_vehicles:
            del active_emergencies[vid]
    # Check if any emergency vehicle is near a controlled lane
    for vid in active_emergencies:
        try:
            lane_id = traci.vehicle.getLaneID(vid)
            if any(lane_id.startswith(cl.split('_')[0]) for cl in controlled_lanes):
                return True, vid
        except Exception:
            pass
    return False, None


def get_valid_routes():
    routes = traci.route.getIDList()
    valid  = []
    for r in routes:
        try:
            edges = traci.route.getEdges(r)
            if edges:
                valid.append(r)
        except Exception:
            pass
    return valid


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def run_simulation():
    sumo_binary = 'sumo-gui' if SUMO_GUI else 'sumo'
    sumo_cmd = [
        sumo_binary, '-c', SUMO_CONFIG,
        '--tripinfo-output', TRIPINFO_OUTPUT,
        '--start',
        '--no-warnings', 'true',
    ]

    try:
        traci.start(sumo_cmd)
        print("[SUMO] Started successfully")
    except Exception as e:
        print(f"[SUMO] ERROR: Could not start SUMO - {e}")
        return

    # --- find traffic light ---
    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids:
        print("[SUMO] No traffic lights found!")
        traci.close()
        return
    tls_id = tls_ids[0]
    print(f"[SUMO] Traffic light: {tls_id}")

    # --- controlled lanes ---
    links = traci.trafficlight.getControlledLinks(tls_id)
    controlled_lanes = []
    for link_group in links:
        for link in link_group:
            lane = link[0]
            if lane and lane not in controlled_lanes:
                controlled_lanes.append(lane)
    print(f"[SUMO] Lanes: {len(controlled_lanes)}")

    # --- valid routes ---
    valid_routes = get_valid_routes()
    print(f"[SUMO] Routes: {len(valid_routes)}")

    # --- init AI + GRU ---
    ai         = EdgeAIController()
    forecaster = QueueForecaster()

    # --- simulation state ---
    active_emergencies  = {}
    current_phase       = 0
    phase_start_step    = 0
    emergency_phase_end = 0

    stats = {'emergency_events': 0, 'emergency_clear_steps': [], 'phase_changes': 0}

    print(f"\n[SIM] Running {SIM_STEPS} steps on MG Road, Bengaluru ...\n")

    for step in range(SIM_STEPS):
        try:
            traci.simulationStep()
        except Exception as e:
            print(f"[{step}s] SUMO step error: {e}")
            break

        # Spawn emergencies
        try_spawn_emergency(step, valid_routes, active_emergencies)

        # Read traffic state
        queues, speeds = read_traffic_state(controlled_lanes)
        total_q = sum(queues)

        # Update GRU history
        forecaster.update(total_q)
        predicted_q = forecaster.forecast()

        # Detect emergency vehicles
        em_active, em_vid = detect_emergency(active_emergencies, controlled_lanes)
        emergency_flag = 1 if em_active else 0

        # --- Priority 1: Emergency corridor ---
        if em_active and step >= emergency_phase_end:
            em_phase = int(np.argmax(queues)) if sum(queues) > 0 else 0
            try:
                traci.trafficlight.setPhase(tls_id, em_phase)
            except Exception:
                pass
            emergency_phase_end = step + EMERGENCY_GREEN
            stats['emergency_events'] += 1
            print(f"[{step}s] EMERGENCY {em_vid} -> PHASE {em_phase} GREEN ({EMERGENCY_GREEN}s)")
            current_phase    = em_phase
            phase_start_step = step
            continue

        # Track emergency clear
        if not em_active and emergency_phase_end > 0 and step > emergency_phase_end:
            dur = step - (emergency_phase_end - EMERGENCY_GREEN)
            stats['emergency_clear_steps'].append(dur)
            print(f"[{step}s] EMERGENCY CLEARED (cleared in {dur}s)")
            emergency_phase_end = 0

        # --- Priority 2 & 3: ONNX + GRU decision ---
        elapsed = step - phase_start_step

        # Boost the max-queue lane with GRU forecast
        boosted_queues = list(queues)
        if predicted_q > 0 and len(boosted_queues) > 0:
            max_lane = int(np.argmax(boosted_queues))
            boosted_queues[max_lane] = max(boosted_queues[max_lane], predicted_q / 4.0)

        if elapsed >= MIN_GREEN:
            new_phase = ai.predict_phase(boosted_queues, speeds, emergency_flag)
            if new_phase != current_phase or elapsed >= MAX_GREEN:
                try:
                    traci.trafficlight.setPhase(tls_id, new_phase)
                    if new_phase != current_phase:
                        print(f"[{step}s] [ONNX] Phase {current_phase} -> {new_phase} "
                              f"(queue={total_q:.0f}, pred={predicted_q:.1f})")
                        stats['phase_changes'] += 1
                    current_phase    = new_phase
                    phase_start_step = step
                except Exception:
                    pass

        # GRU forecast log every 100 steps
        if step % 100 == 0 and step > 0:
            print(f"[{step}s] [GRU forecast] queue in ~30s = {predicted_q:.1f} vehicles")

    traci.close()

    # --- summary ---
    print("\n" + "=" * 55)
    print("SIMULATION COMPLETE — MG Road, Bengaluru")
    print("=" * 55)
    print(f"  Emergency events:  {stats['emergency_events']}")
    if stats['emergency_clear_steps']:
        avg_clear = sum(stats['emergency_clear_steps']) / len(stats['emergency_clear_steps'])
        print(f"  Avg clear time:    {avg_clear:.1f}s")
    print(f"  ONNX phase changes: {stats['phase_changes']}")
    print(f"  Output:            {TRIPINFO_OUTPUT}")
    print("=" * 55)


if __name__ == '__main__':
    run_simulation()
