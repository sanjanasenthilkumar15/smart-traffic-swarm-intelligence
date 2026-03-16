"""
EDGE AI TRAFFIC CONTROLLER - Enhanced Emergency Vehicle Demo

This version includes:
- MORE emergency vehicles (higher frequency)
- DIVERSE emergency types with unique colors
- CLEAR visual signal changes
- ENHANCED console output with emojis
"""

import traci
import sys
import random
import numpy as np
import pickle
from typing import List, Dict, Tuple

# ONNX Runtime for edge inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
SUMO_CONFIG = "kpit.sumocfg"
SUMO_GUI = True
SIM_STEPS = 5000

# AI Model files
ONNX_MODEL = "traffic_model.onnx"
SCALER_FILE = "traffic_scaler.pkl"
CONFIG_FILE = "model_config.pkl"

# Safety constraints
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 60
EMERGENCY_GREEN_TIME = 45

# ============================================================================
# ENHANCED EMERGENCY VEHICLE CONFIGURATION
# ============================================================================

# More emergency vehicle types with distinct colors
EMERGENCY_VEHICLES = {
    "ambulance": {
        "color": (255, 255, 255, 255),  # White
        "icon": "🚑",
        "priority": 1,  # Highest
        "max_speed": 30
    },
    "fire_truck": {
        "color": (255, 0, 0, 255),  # Red
        "icon": "🚒",
        "priority": 1,  # Highest
        "max_speed": 28
    },
    "police": {
        "color": (0, 0, 255, 255),  # Blue
        "icon": "🚓",
        "priority": 2,
        "max_speed": 32
    },
    "rescue": {
        "color": (255, 165, 0, 255),  # Orange
        "icon": "🚐",
        "priority": 2,
        "max_speed": 26
    },
    "paramedic": {
        "color": (0, 255, 0, 255),  # Green
        "icon": "🚙",
        "priority": 1,
        "max_speed": 29
    },
    "emergency_doctor": {
        "color": (255, 255, 0, 255),  # Yellow
        "icon": "🚕",
        "priority": 1,
        "max_speed": 27
    },
    "police_swat": {
        "color": (0, 0, 128, 255),  # Dark Blue
        "icon": "🚔",
        "priority": 1,
        "max_speed": 31
    },
    "fire_chief": {
        "color": (139, 0, 0, 255),  # Dark Red
        "icon": "🚗",
        "priority": 2,
        "max_speed": 28
    }
}

# INCREASED spawn frequency for demonstration
EMERGENCY_SPAWN_INTERVAL = 80      # Reduced from 200 (spawn more often!)
EMERGENCY_SPAWN_PROBABILITY = 0.6  # Increased from 0.3 (60% chance!)

# Multiple vehicles can spawn at once
MULTI_SPAWN_PROBABILITY = 0.4      # 40% chance to spawn 2-3 vehicles at once
MAX_SIMULTANEOUS_SPAWN = 3

# ============================================================================
# AI INFERENCE ENGINE
# ============================================================================
class EdgeAIController:
    """Edge AI inference engine for traffic signal control"""
    
    def __init__(self):
        self.onnx_session = None
        self.scaler = None
        self.config = None
        self.fallback_mode = False
        self.inference_count = 0
        
    def initialize(self):
        """Load ONNX model and preprocessor"""
        
        print("\n" + "=" * 80)
        print("INITIALIZING EDGE AI CONTROLLER")
        print("=" * 80)
        
        if not ONNX_AVAILABLE:
            print("✗ ONNX Runtime not available - using fallback mode")
            self.fallback_mode = True
            return False
        
        try:
            with open(CONFIG_FILE, 'rb') as f:
                self.config = pickle.load(f)
            print(f"✓ Model config loaded: {self.config['num_phases']} phases")
        except:
            self.fallback_mode = True
            return False
        
        try:
            with open(SCALER_FILE, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Feature scaler loaded")
        except:
            self.fallback_mode = True
            return False
        
        try:
            self.onnx_session = ort.InferenceSession(ONNX_MODEL)
            
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            
            print(f"✓ ONNX model loaded: {ONNX_MODEL}")
            print(f"    Input: {input_name}")
            print(f"    Output: {output_name}")
            print(f"    Ready for edge inference")
            
        except Exception as e:
            print(f"✗ Failed to load ONNX model: {e}")
            self.fallback_mode = True
            return False
        
        print("=" * 80 + "\n")
        return True
    
    def predict_phase(self, traffic_state: Dict) -> int:
        """AI inference: predict optimal traffic phase"""
        
        if self.fallback_mode or self.onnx_session is None:
            return self._rule_based_fallback(traffic_state)
        
        try:
            features = (
                traffic_state['queues'] + 
                traffic_state['speeds'] + 
                [traffic_state['emergency_flag']]
            )
            
            features_array = np.array([features], dtype=np.float32)
            features_scaled = self.scaler.transform(features_array)
            
            input_name = self.onnx_session.get_inputs()[0].name
            ort_inputs = {input_name: features_scaled.astype(np.float32)}
            ort_outputs = self.onnx_session.run(None, ort_inputs)
            
            predicted_phase = int(np.argmax(ort_outputs[0], axis=1)[0])
            self.inference_count += 1
            
            return predicted_phase
            
        except Exception as e:
            print(f"⚠ AI inference failed: {e}")
            return self._rule_based_fallback(traffic_state)
    
    def _rule_based_fallback(self, state: Dict) -> int:
        """Fallback rule-based control"""
        current_phase = state['current_phase']
        total_queue = sum(state['queues'])
        
        if state['emergency_flag'] == 1:
            return current_phase
        
        if total_queue >= 10:
            num_phases = self.config['num_phases'] if self.config else 3
            return (current_phase + 1) % num_phases
        
        return current_phase

# ============================================================================
# SYSTEM STATE MANAGER
# ============================================================================
class TrafficSystemState:
    """Manages complete system state"""
    
    def __init__(self):
        self.emergency_mode = False
        self.emergency_start_time = 0
        self.last_emergency_spawn = 0
        self.last_phase_change = 0
        self.current_phase_duration = 0
        self.phase_changes = 0
        self.emergency_vehicles_served = 0
        self.emergency_by_type = {}  # Track each type

# ============================================================================
# TRAFFIC STATE EXTRACTION
# ============================================================================
def extract_traffic_state(tls_id: str, controlled_lanes: List[str]) -> Dict:
    """Extract real-time traffic state from SUMO"""
    
    lanes_to_monitor = controlled_lanes[:4]
    while len(lanes_to_monitor) < 4:
        if lanes_to_monitor:
            lanes_to_monitor.append(lanes_to_monitor[-1])
        else:
            lanes_to_monitor.append("dummy")
    
    queues = []
    speeds = []
    
    for lane in lanes_to_monitor:
        try:
            queue = traci.lane.getLastStepHaltingNumber(lane)
            speed = traci.lane.getLastStepMeanSpeed(lane)
            queues.append(queue)
            speeds.append(round(speed, 2))
        except:
            queues.append(0)
            speeds.append(0.0)
    
    # Enhanced emergency detection with type information
    emergency_flag = 0
    emergency_vehicles = []
    emergency_details = []
    
    try:
        all_vehicles = traci.vehicle.getIDList()
        for veh_id in all_vehicles:
            # Check if it's an emergency vehicle
            for veh_type in EMERGENCY_VEHICLES.keys():
                if veh_type in veh_id.lower():
                    try:
                        veh_lane = traci.vehicle.getLaneID(veh_id)
                        if veh_lane in controlled_lanes or any(veh_lane.startswith(cl.split('_')[0]) for cl in controlled_lanes):
                            emergency_flag = 1
                            emergency_vehicles.append(veh_id)
                            emergency_details.append({
                                'id': veh_id,
                                'type': veh_type,
                                'icon': EMERGENCY_VEHICLES[veh_type]['icon'],
                                'lane': veh_lane
                            })
                    except:
                        pass
                    break
    except:
        pass
    
    try:
        current_phase = traci.trafficlight.getPhase(tls_id)
    except:
        current_phase = 0
    
    return {
        'queues': queues,
        'speeds': speeds,
        'emergency_flag': emergency_flag,
        'current_phase': current_phase,
        'emergency_vehicles': emergency_vehicles,
        'emergency_details': emergency_details
    }

# ============================================================================
# GET NUMBER OF PHASES
# ============================================================================
def get_num_phases(tls_id: str) -> int:
    """Get number of phases (compatible with all SUMO versions)"""
    try:
        return traci.trafficlight.getPhaseNumber(tls_id)
    except AttributeError:
        try:
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            return len(logic.phases)
        except:
            return 3

# ============================================================================
# SAFETY CONSTRAINTS
# ============================================================================
def apply_safety_constraints(
    predicted_phase: int,
    current_phase: int,
    time_in_phase: float,
    emergency_mode: bool,
    num_phases: int
) -> Tuple[int, bool]:
    """Safety layer: validate AI decision"""
    
    if emergency_mode:
        return current_phase, False
    
    if time_in_phase < MIN_GREEN_TIME and predicted_phase != current_phase:
        return current_phase, False
    
    if time_in_phase > MAX_GREEN_TIME:
        next_phase = (current_phase + 1) % num_phases
        return next_phase, True
    
    if predicted_phase < 0 or predicted_phase >= num_phases:
        return current_phase, False
    
    return predicted_phase, True

# ============================================================================
# CONTROL ACTUATION
# ============================================================================
def execute_phase_change(tls_id: str, target_phase: int, state: TrafficSystemState):
    """Execute traffic signal phase change"""
    try:
        current_phase = traci.trafficlight.getPhase(tls_id)
        
        if target_phase != current_phase:
            traci.trafficlight.setPhase(tls_id, target_phase)
            state.phase_changes += 1
            state.last_phase_change = traci.simulation.getTime()
            state.current_phase_duration = 0
            
            # Visual indicator
            print(f"    🚦 SIGNAL CHANGED: Phase {current_phase} → {target_phase}")
            return True
        return False
        
    except Exception as e:
        print(f"⚠ Phase change failed: {e}")
        return False

# ============================================================================
# ENHANCED EMERGENCY VEHICLE SPAWNING
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

def try_spawn_emergency(step: int, state: TrafficSystemState):
    """
    Enhanced emergency vehicle spawning with:
    - Multiple vehicle types
    - Higher spawn frequency
    - Unique colors for each type
    - Multiple simultaneous spawns
    """
    
    if step - state.last_emergency_spawn < EMERGENCY_SPAWN_INTERVAL:
        return
    
    if random.random() > EMERGENCY_SPAWN_PROBABILITY:
        state.last_emergency_spawn = step
        return
    
    routes = get_valid_routes()
    if not routes:
        return
    
    # Determine how many vehicles to spawn
    num_to_spawn = 1
    if random.random() < MULTI_SPAWN_PROBABILITY:
        num_to_spawn = random.randint(2, MAX_SIMULTANEOUS_SPAWN)
    
    spawned = []
    
    for i in range(num_to_spawn):
        route = random.choice(routes)
        veh_type = random.choice(list(EMERGENCY_VEHICLES.keys()))
        vid = f"{veh_type}_{step}_{i}"
        
        try:
            traci.vehicle.add(vehID=vid, routeID=route, typeID="DEFAULT_VEHTYPE")
            
            # Set unique color for this vehicle type
            veh_config = EMERGENCY_VEHICLES[veh_type]
            traci.vehicle.setColor(vid, veh_config['color'])
            traci.vehicle.setMaxSpeed(vid, veh_config['max_speed'])
            
            # Make it stand out
            traci.vehicle.setLength(vid, 6.0)  # Slightly larger
            
            spawned.append({
                'id': vid,
                'type': veh_type,
                'icon': veh_config['icon']
            })
            
            # Track by type
            if veh_type not in state.emergency_by_type:
                state.emergency_by_type[veh_type] = 0
            state.emergency_by_type[veh_type] += 1
            
        except Exception as e:
            pass
    
    if spawned:
        state.last_emergency_spawn = step
        
        # Enhanced console output
        print(f"\n{'='*60}")
        print(f"[{step}s] 🚨 EMERGENCY VEHICLES SPAWNED!")
        print(f"{'='*60}")
        for vehicle in spawned:
            print(f"  {vehicle['icon']} {vehicle['type'].upper().replace('_', ' ')} → {vehicle['id']}")
        print(f"{'='*60}\n")

# ============================================================================
# LANE EXTRACTION
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
# MAIN SIMULATION LOOP
# ============================================================================
def run_edge_ai_simulation():
    """Main simulation with Enhanced Emergency Vehicle Demo"""
    
    print("\n" + "=" * 80)
    print("🚦 EDGE AI TRAFFIC CONTROL SYSTEM - ENHANCED EMERGENCY DEMO 🚦")
    print("=" * 80)
    print("Features:")
    print("  • 8 Different Emergency Vehicle Types")
    print("  • Unique Colors for Each Type")
    print("  • High Spawn Frequency (60%)")
    print("  • Multiple Simultaneous Spawns")
    print("  • Clear Visual Signal Changes")
    print("=" * 80)
    
    # Initialize AI controller
    ai_controller = EdgeAIController()
    if not ai_controller.initialize():
        print("\n⚠ WARNING: Running in FALLBACK mode")
    
    system_state = TrafficSystemState()
    
    # Start SUMO
    print("\n" + "=" * 80)
    print("STARTING SUMO SIMULATION")
    print("=" * 80)
    
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
        print(f"✓ Controlling traffic light: {tls_id}")
        
        controlled_lanes = get_controlled_lanes(tls_id)
        print(f"✓ Monitoring {len(controlled_lanes)} lanes")
        
        num_phases = get_num_phases(tls_id)
        print(f"✓ Traffic light has {num_phases} phases")
        
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        traci.close()
        sys.exit(1)
    
    print(f"\n▶ Starting ENHANCED simulation for {SIM_STEPS} seconds...")
    print("   Watch for emergency vehicles with different colors!")
    print("=" * 80 + "\n")
    
    # Main control loop
    step = 0
    last_status_print = 0
    
    try:
        while step < SIM_STEPS:
            traci.simulationStep()
            step += 1
            current_time = traci.simulation.getTime()
            
            if traci.simulation.getMinExpectedNumber() == 0:
                print(f"\n✓ No more vehicles at step {step}")
                break
            
            # Spawn emergency vehicles (enhanced)
            try_spawn_emergency(step, system_state)
            
            # PERCEPTION
            traffic_state = extract_traffic_state(tls_id, controlled_lanes)
            
            # EMERGENCY OVERRIDE with enhanced display
            if traffic_state['emergency_flag'] == 1:
                if not system_state.emergency_mode:
                    system_state.emergency_mode = True
                    system_state.emergency_start_time = current_time
                    system_state.emergency_vehicles_served += 1
                    
                    print(f"\n{'#'*60}")
                    print(f"[{step}s] 🚨🚨 EMERGENCY MODE ACTIVATED 🚨🚨")
                    print(f"{'#'*60}")
                    
                    for detail in traffic_state['emergency_details']:
                        print(f"  {detail['icon']} {detail['type'].upper().replace('_', ' ')}")
                        print(f"      Vehicle ID: {detail['id']}")
                        print(f"      Lane: {detail['lane']}")
                    
                    print(f"\n  🚦 HOLDING GREEN SIGNAL FOR {EMERGENCY_GREEN_TIME}s")
                    print(f"  ⏱️  Current Phase: {traffic_state['current_phase']} (LOCKED)")
                    print(f"{'#'*60}\n")
                
                continue
            
            # Clear emergency mode with notification
            if system_state.emergency_mode and traffic_state['emergency_flag'] == 0:
                system_state.emergency_mode = False
                print(f"\n{'='*60}")
                print(f"[{step}s] ✓ EMERGENCY CLEARED")
                print(f"{'='*60}")
                print(f"  Resuming AI control...")
                print(f"  Total emergencies served: {system_state.emergency_vehicles_served}")
                print(f"{'='*60}\n")
            
            # AI INFERENCE
            predicted_phase = ai_controller.predict_phase(traffic_state)
            
            time_in_phase = current_time - system_state.last_phase_change
            system_state.current_phase_duration = time_in_phase
            
            # SAFETY LAYER
            approved_phase, can_change = apply_safety_constraints(
                predicted_phase=predicted_phase,
                current_phase=traffic_state['current_phase'],
                time_in_phase=time_in_phase,
                emergency_mode=system_state.emergency_mode,
                num_phases=num_phases
            )
            
            # ACTUATION
            if can_change and approved_phase != traffic_state['current_phase']:
                changed = execute_phase_change(tls_id, approved_phase, system_state)
                
                if changed and not ai_controller.fallback_mode:
                    mode = "🤖 AI"
                    print(f"[{step}s] {mode}: Phase {traffic_state['current_phase']} → {approved_phase} "
                          f"(Queue: {sum(traffic_state['queues'])}, Time: {time_in_phase:.1f}s)")
            
            # Enhanced status updates
            if step - last_status_print >= 500:
                num_vehicles = len(traci.vehicle.getIDList())
                mode = "AI" if not ai_controller.fallback_mode else "FALLBACK"
                
                print(f"\n{'─'*60}")
                print(f"[{step}s] 📊 STATUS REPORT ({mode} Mode)")
                print(f"{'─'*60}")
                print(f"  🚗 Total Vehicles: {num_vehicles}")
                print(f"  🚦 Phase Changes: {system_state.phase_changes}")
                print(f"  🚨 Emergencies Served: {system_state.emergency_vehicles_served}")
                
                if system_state.emergency_by_type:
                    print(f"\n  Emergency Breakdown:")
                    for veh_type, count in sorted(system_state.emergency_by_type.items()):
                        icon = EMERGENCY_VEHICLES[veh_type]['icon']
                        print(f"    {icon} {veh_type.replace('_', ' ').title()}: {count}")
                
                if not ai_controller.fallback_mode:
                    print(f"\n  🤖 AI Inferences: {ai_controller.inference_count}")
                
                print(f"{'─'*60}\n")
                last_status_print = step
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        traci.close()
        
        # Enhanced summary
        print("\n" + "=" * 80)
        print("📊 FINAL SIMULATION SUMMARY")
        print("=" * 80)
        print(f"Duration: {step} seconds")
        print(f"Control Mode: {'🤖 AI (ONNX)' if not ai_controller.fallback_mode else '📋 Rule-based fallback'}")
        print(f"Phase Changes: {system_state.phase_changes}")
        print(f"Total Emergency Vehicles: {system_state.emergency_vehicles_served}")
        
        if system_state.emergency_by_type:
            print(f"\nEmergency Vehicles by Type:")
            for veh_type, count in sorted(system_state.emergency_by_type.items(), key=lambda x: x[1], reverse=True):
                icon = EMERGENCY_VEHICLES[veh_type]['icon']
                print(f"  {icon} {veh_type.replace('_', ' ').title()}: {count}")
        
        if not ai_controller.fallback_mode:
            print(f"\nAI Inferences: {ai_controller.inference_count}")
        
        print("=" * 80 + "\n")

if __name__ == "__main__":
    run_edge_ai_simulation()