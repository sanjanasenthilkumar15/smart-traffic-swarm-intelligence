"""
DATA LOGGER - Traffic State Extraction & Dataset Generation

This module extracts real-time traffic state from SUMO via TraCI and logs
state-action pairs for supervised learning.

Input: SUMO simulation via TraCI
Output: CSV dataset with traffic states and control decisions
"""

import traci
import sys
import csv
import random
from typing import List, Dict, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================
SUMO_CONFIG = "kpit.sumocfg"
SUMO_GUI = True
SIM_STEPS = 5000

# Dataset output
DATASET_FILE = "traffic_training_data.csv"

# Emergency vehicle spawning
EMERGENCY_TYPES = ["emergency", "ambulance", "fire", "police"]
EMERGENCY_SPAWN_INTERVAL = 200
EMERGENCY_SPAWN_PROBABILITY = 0.3

# Control parameters
CONGESTION_THRESHOLD = 10
MIN_GREEN = 10
MAX_GREEN_EXTENSION = 60

# ============================================================================
# STATE TRACKING
# ============================================================================
class DataLogger:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.csv_file = None
        self.writer = None
        self.last_emergency_spawn = 0
        self.samples_collected = 0
        
    def initialize(self):
        """Initialize CSV file with header"""
        self.csv_file = open(self.output_file, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        
        # Header: 4 queues + 4 speeds + emergency flag + current phase + label (next phase)
        header = [
            "queue_lane_0", "queue_lane_1", "queue_lane_2", "queue_lane_3",
            "avg_speed_lane_0", "avg_speed_lane_1", "avg_speed_lane_2", "avg_speed_lane_3",
            "emergency_flag",
            "current_phase",
            "next_phase"  # This is the LABEL
        ]
        self.writer.writerow(header)
        print(f"✓ Dataset initialized: {self.output_file}")
        
    def close(self):
        """Close CSV file"""
        if self.csv_file:
            self.csv_file.close()
            print(f"✓ Dataset saved: {self.samples_collected} samples collected")

# ============================================================================
# TRAFFIC STATE EXTRACTION
# ============================================================================
def extract_traffic_state(tls_id: str, controlled_lanes: List[str]) -> Dict:
    """
    Extract structured traffic state from SUMO.
    
    This is the EXACT state vector the AI model will use.
    
    Returns:
        Dict with keys:
        - queues: List[int] - vehicles halting per lane
        - speeds: List[float] - average speed per lane
        - emergency_flag: int (0 or 1)
        - current_phase: int
    """
    # Limit to 4 lanes for model simplicity (can be extended)
    lanes_to_monitor = controlled_lanes[:4]
    
    # Pad if less than 4 lanes
    while len(lanes_to_monitor) < 4:
        lanes_to_monitor.append(lanes_to_monitor[-1] if lanes_to_monitor else "dummy")
    
    queues = []
    speeds = []
    
    for lane in lanes_to_monitor:
        try:
            queue = traci.lane.getLastStepHaltingNumber(lane)
            speed = traci.lane.getLastStepMeanSpeed(lane)
            queues.append(queue)
            speeds.append(round(speed, 2))
        except:
            # Fallback for invalid lanes
            queues.append(0)
            speeds.append(0.0)
    
    # Check for emergency vehicles
    emergency_flag = 0
    try:
        all_vehicles = traci.vehicle.getIDList()
        for veh_id in all_vehicles:
            if any(etype in veh_id.lower() for etype in EMERGENCY_TYPES):
                emergency_flag = 1
                break
    except:
        pass
    
    # Get current phase
    try:
        current_phase = traci.trafficlight.getPhase(tls_id)
    except:
        current_phase = 0
    
    return {
        'queues': queues,
        'speeds': speeds,
        'emergency_flag': emergency_flag,
        'current_phase': current_phase
    }

# ============================================================================
# RULE-BASED DECISION LOGIC (GENERATES LABELS)
# ============================================================================
def rule_based_decision(state: Dict, tls_id: str, num_phases: int) -> int:
    """
    Rule-based logic that generates training labels.
    
    This is what the AI will learn to mimic (and eventually improve upon).
    
    Decision rules:
    1. Emergency detected → Keep current phase (priority)
    2. High congestion → Rotate to next phase
    3. Normal traffic → Keep current phase
    
    Returns:
        next_phase: int - the phase to switch to
    """
    current_phase = state['current_phase']
    total_queue = sum(state['queues'])
    emergency = state['emergency_flag']
    
    # Rule 1: Emergency priority - keep current phase
    if emergency == 1:
        return current_phase
    
    # Rule 2: Congestion - rotate phase
    if total_queue >= CONGESTION_THRESHOLD:
        # Switch to next green phase
        next_phase = (current_phase + 1) % num_phases
        return next_phase
    
    # Rule 3: Normal traffic - maintain current phase
    return current_phase

# ============================================================================
# EMERGENCY VEHICLE SPAWNING
# ============================================================================
def get_valid_routes() -> List[str]:
    """Get valid routes from simulation"""
    try:
        routes = traci.route.getIDList()
        if routes:
            return list(routes)
        
        vehicles = traci.vehicle.getIDList()
        valid_routes = set()
        for v in vehicles:
            try:
                route_id = traci.vehicle.getRouteID(v)
                valid_routes.add(route_id)
            except:
                pass
        
        return list(valid_routes) if valid_routes else []
    except:
        return []

def try_spawn_emergency(step: int, logger: DataLogger):
    """Spawn emergency vehicles for realistic scenarios"""
    if step - logger.last_emergency_spawn < EMERGENCY_SPAWN_INTERVAL:
        return
    
    if random.random() > EMERGENCY_SPAWN_PROBABILITY:
        logger.last_emergency_spawn = step
        return
    
    routes = get_valid_routes()
    if not routes:
        return
    
    route = random.choice(routes)
    etype = random.choice(["ambulance", "fire", "police"])
    vid = f"{etype}_{step}"
    
    try:
        traci.vehicle.add(
            vehID=vid,
            routeID=route,
            typeID="DEFAULT_VEHTYPE"
        )
        
        colors = {
            "ambulance": (255, 255, 255, 255),
            "fire": (255, 0, 0, 255),
            "police": (0, 0, 255, 255)
        }
        traci.vehicle.setColor(vid, colors.get(etype, (255, 255, 255, 255)))
        traci.vehicle.setMaxSpeed(vid, 25)
        
        logger.last_emergency_spawn = step
        
    except:
        pass

# ============================================================================
# LANE EXTRACTION HELPER
# ============================================================================
def get_controlled_lanes(tls_id: str) -> List[str]:
    """Get lanes controlled by traffic light"""
    try:
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        lanes = set()
        
        for link_list in controlled_links:
            for link in link_list:
                lanes.add(link[0])
        
        return list(lanes)
    except:
        return []

# ============================================================================
# MAIN DATA COLLECTION LOOP
# ============================================================================
def collect_training_data():
    """
    Main data collection simulation.
    
    Workflow:
    1. Run SUMO simulation
    2. Extract traffic state every step
    3. Apply rule-based decision logic
    4. Log (state, action) pair to CSV
    5. Execute action in simulation
    """
    
    print("=" * 80)
    print("DATA LOGGER - Generating AI Training Dataset")
    print("=" * 80)
    
    # Initialize logger
    logger = DataLogger(DATASET_FILE)
    logger.initialize()
    
    # Start SUMO
    sumo_binary = "sumo-gui" if SUMO_GUI else "sumo"
    sumo_cmd = [sumo_binary, "-c", SUMO_CONFIG, "--start", "--quit-on-end"]
    
    try:
        traci.start(sumo_cmd)
        print("✓ SUMO started successfully")
    except Exception as e:
        print(f"✗ Failed to start SUMO: {e}")
        sys.exit(1)
    
    # Get traffic light info
    try:
        tls_list = traci.trafficlight.getIDList()
        if not tls_list:
            print("✗ No traffic lights found")
            traci.close()
            sys.exit(1)
        
        tls_id = tls_list[0]
        print(f"✓ Monitoring traffic light: {tls_id}")
        
        controlled_lanes = get_controlled_lanes(tls_id)
        print(f"✓ Monitoring {len(controlled_lanes)} lanes")
        
        # Get number of phases
        num_phases = traci.trafficlight.getPhase(tls_id)
        print(f"✓ Traffic light has {num_phases} phases")
        
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        traci.close()
        sys.exit(1)
    
    print(f"\n▶ Starting data collection for {SIM_STEPS} steps...")
    print(f"  Emergency vehicles will spawn randomly\n")
    
    # Main simulation loop
    step = 0
    try:
        while step < SIM_STEPS:
            traci.simulationStep()
            step += 1
            
            # Check if simulation is done
            if traci.simulation.getMinExpectedNumber() == 0:
                print(f"\n✓ No more vehicles at step {step}")
                break
            
            # Spawn emergency vehicles
            try_spawn_emergency(step, logger)
            
            # EXTRACT STATE
            state = extract_traffic_state(tls_id, controlled_lanes)
            
            # APPLY RULE-BASED LOGIC (generates label)
            next_phase = rule_based_decision(state, tls_id, num_phases)
            
            # LOG TO DATASET
            row = (
                state['queues'] + 
                state['speeds'] + 
                [state['emergency_flag']] +
                [state['current_phase']] +
                [next_phase]  # LABEL
            )
            logger.writer.writerow(row)
            logger.samples_collected += 1
            
            # EXECUTE DECISION (optional - for realistic simulation)
            # This makes the training data more realistic
            if next_phase != state['current_phase']:
                try:
                    traci.trafficlight.setPhase(tls_id, next_phase)
                except:
                    pass
            
            # Status updates
            if step % 500 == 0:
                print(f"[{step}s] Collected {logger.samples_collected} samples, "
                      f"Queue: {sum(state['queues'])}, "
                      f"Emergency: {state['emergency_flag']}")
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        traci.close()
        logger.close()
        print(f"\n✓ Data collection completed")
        print(f"✓ Dataset: {DATASET_FILE}")
        print("=" * 80)

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    collect_training_data()
