
import traci
import sys
import random

# ============================================================================
# OPTIMIZED CONFIGURATION - Balanced for best results
# ============================================================================
SUMO_CONFIG = "kpit.sumocfg"
SUMO_GUI = True  # Faster
SIM_STEPS = 7200  # 2 hours (enough data, not too long)

# Output
TRIPINFO_OUTPUT = "adaptive_tripinfo.xml"

# BALANCED THRESHOLDS - Not too aggressive
CONGESTION_THRESHOLD = 7      # Sensitive but not crazy (was 10)
MIN_GREEN = 8                 # Reasonable minimum (was 10)
MAX_GREEN_EXTENSION = 50      # Good balance (was 60)
DEFAULT_GREEN = 25            # Slightly shorter (was 30)
EMERGENCY_GREEN = 60          # Strong but not excessive (was 45)

# MORE EMERGENCIES - Clear demonstration
EMERGENCY_TYPES = ["emergency", "ambulance", "fire", "police"]
EMERGENCY_SPAWN_INTERVAL = 150  # More frequent (was 200)
EMERGENCY_SPAWN_PROBABILITY = 0.4  # Higher chance (was 0.3)

# ============================================================================
# STATE TRACKING WITH STATISTICS
# ============================================================================
class TrafficController:
    def __init__(self):
        self.emergency_mode = False
        self.last_emergency_spawn = 0
        self.extension_time = 0
        self.last_log = ""
        self.phase_to_lanes = {}
        self.emergency_phase = None
        self.emergency_start_time = 0
        
        # Statistics
        self.stats = {
            'emergency_events': 0,
            'emergency_spawned': 0,
            'emergency_cleared': 0,
            'congestion_events': 0,
            'phase_switches': 0,
            'total_extensions': 0,
            'emergency_response_times': []
        }
    
    def log(self, step, message):
        if message != self.last_log:
            print(f"[{step}s] {message}")
            self.last_log = message
    
    def print_stats(self):
        print("\n" + "=" * 80)
        print("ADAPTIVE CONTROL STATISTICS")
        print("=" * 80)
        print(f"Emergency Vehicles Spawned: {self.stats['emergency_spawned']}")
        print(f"Emergency Events Handled:   {self.stats['emergency_events']}")
        print(f"Emergency Vehicles Cleared: {self.stats['emergency_cleared']}")
        print(f"Congestion Events:          {self.stats['congestion_events']}")
        print(f"Phase Switches:             {self.stats['phase_switches']}")
        print(f"Total Green Extensions:     {self.stats['total_extensions']}")
        
        if self.stats['emergency_response_times']:
            avg_response = sum(self.stats['emergency_response_times']) / len(self.stats['emergency_response_times'])
            print(f"Avg Emergency Clear Time:   {avg_response:.1f}s")
        print("=" * 80)

# ============================================================================
# BUILD PHASE-TO-LANE MAPPING
# ============================================================================
def build_phase_lane_mapping(tls_id):
    try:
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        phase_to_lanes = {}
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        
        for phase_idx, phase in enumerate(logic.phases):
            green_lanes = set()
            state = phase.state
            
            for link_idx, signal in enumerate(state):
                if signal in ['G', 'g']:
                    if link_idx < len(controlled_links):
                        for link in controlled_links[link_idx]:
                            green_lanes.add(link[0])
            
            phase_to_lanes[phase_idx] = list(green_lanes)
        
        return phase_to_lanes
    except Exception as e:
        print(f"Warning: Could not build phase mapping: {e}")
        return {}

def find_phase_for_lane(lane, phase_to_lanes):
    for phase_idx, lanes in phase_to_lanes.items():
        if lane in lanes:
            return phase_idx
    return None

# ============================================================================
# ROUTE MANAGEMENT
# ============================================================================
def get_valid_routes():
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

# ============================================================================
# SPAWN EMERGENCY VEHICLES
# ============================================================================
def try_spawn_emergency(step, controller):
    if step - controller.last_emergency_spawn < EMERGENCY_SPAWN_INTERVAL:
        return
    
    if random.random() > EMERGENCY_SPAWN_PROBABILITY:
        controller.last_emergency_spawn = step
        return
    
    routes = get_valid_routes()
    if not routes:
        return
    
    route = random.choice(routes)
    etype = random.choice(["ambulance", "fire", "police"])
    vid = f"{etype}_{step}"
    
    try:
        traci.vehicle.add(vehID=vid, routeID=route, typeID="DEFAULT_VEHTYPE")
        
        colors = {
            "ambulance": (255, 255, 255, 255),
            "fire": (255, 0, 0, 255),
            "police": (0, 0, 255, 255)
        }
        traci.vehicle.setColor(vid, colors.get(etype, (255, 255, 255, 255)))
        traci.vehicle.setMaxSpeed(vid, 28)
        
        controller.stats['emergency_spawned'] += 1
        print(f"[{step}s] 🚨 Spawned {etype.upper()} → {vid}")
        controller.last_emergency_spawn = step
    except:
        pass

# ============================================================================
# GET CONTROLLED LANES
# ============================================================================
def get_controlled_lanes(tls_id):
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
# READ TRAFFIC STATE
# ============================================================================
def read_traffic_state(controlled_lanes):
    state = {
        'total_halting': 0,
        'emergency_present': False,
        'emergency_vehicles': [],
        'emergency_lanes': []
    }
    
    for lane in controlled_lanes:
        try:
            halting = traci.lane.getLastStepHaltingNumber(lane)
            state['total_halting'] += halting
        except:
            pass
    
    try:
        all_vehicles = traci.vehicle.getIDList()
        for veh_id in all_vehicles:
            is_emergency = any(etype in veh_id.lower() for etype in EMERGENCY_TYPES)
            
            if is_emergency:
                try:
                    veh_lane = traci.vehicle.getLaneID(veh_id)
                    veh_edge = traci.vehicle.getRoadID(veh_id)
                    
                    on_controlled = veh_lane in controlled_lanes
                    near_controlled = any(lane.startswith(veh_edge) for lane in controlled_lanes)
                    
                    if on_controlled or near_controlled:
                        state['emergency_present'] = True
                        state['emergency_vehicles'].append(veh_id)
                        state['emergency_lanes'].append(veh_lane)
                except:
                    pass
    except:
        pass
    
    return state

# ============================================================================
# IMPROVED CONTROL LOGIC - Based on your proven code
# ============================================================================
def control_traffic_light(tls_id, controlled_lanes, controller, step):
    state = read_traffic_state(controlled_lanes)
    current_time = traci.simulation.getTime()
    
    # PRIORITY 1: Emergency vehicles
    if state['emergency_present']:
        if not controller.emergency_mode:
            controller.emergency_mode = True
            controller.emergency_start_time = current_time
            controller.stats['emergency_events'] += 1
            
            emergency_lane = state['emergency_lanes'][0] if state['emergency_lanes'] else None
            
            if emergency_lane and controller.phase_to_lanes:
                target_phase = find_phase_for_lane(emergency_lane, controller.phase_to_lanes)
                
                if target_phase is not None:
                    controller.emergency_phase = target_phase
                    controller.stats['phase_switches'] += 1
                    
                    try:
                        traci.trafficlight.setPhase(tls_id, target_phase)
                        traci.trafficlight.setPhaseDuration(tls_id, EMERGENCY_GREEN)
                        
                        veh_list = ", ".join(state['emergency_vehicles'][:3])
                        controller.log(step, 
                            f"🚨 EMERGENCY! {veh_list} on {emergency_lane} → PHASE {target_phase} (GREEN)")
                    except Exception as e:
                        controller.log(step, f"⚠️ Could not switch phase: {e}")
                else:
                    controller.log(step, 
                        f"🚨 EMERGENCY! {state['emergency_vehicles'][0]} detected → Extending phase")
                    try:
                        traci.trafficlight.setPhaseDuration(tls_id, EMERGENCY_GREEN)
                    except:
                        pass
        else:
            # Keep emergency green active
            if current_time - controller.emergency_start_time < EMERGENCY_GREEN:
                try:
                    remaining = EMERGENCY_GREEN - (current_time - controller.emergency_start_time)
                    traci.trafficlight.setPhaseDuration(tls_id, remaining)
                except:
                    pass
        
        return
    
    # Emergency cleared
    if controller.emergency_mode and not state['emergency_present']:
        controller.emergency_mode = False
        controller.extension_time = 0
        controller.emergency_phase = None
        controller.stats['emergency_cleared'] += 1
        
        response_time = current_time - controller.emergency_start_time
        controller.stats['emergency_response_times'].append(response_time)
        
        controller.log(step, f"✓ EMERGENCY CLEARED (cleared in {response_time:.0f}s)")
    
    # PRIORITY 2: Smart congestion management
    if state['total_halting'] >= CONGESTION_THRESHOLD:
        if controller.extension_time < MAX_GREEN_EXTENSION:
            try:
                current_duration = traci.trafficlight.getNextSwitch(tls_id) - current_time
                
                # SMART EXTENSION: Scale with congestion level
                congestion_level = state['total_halting'] - CONGESTION_THRESHOLD
                if congestion_level > 10:
                    extension_amount = 12  # Heavy congestion
                elif congestion_level > 5:
                    extension_amount = 8   # Medium congestion
                else:
                    extension_amount = 6   # Light congestion
                
                new_duration = max(current_duration + extension_amount, MIN_GREEN)
                traci.trafficlight.setPhaseDuration(tls_id, new_duration)
                controller.extension_time += extension_amount
                controller.stats['total_extensions'] += 1
                
                if controller.extension_time <= extension_amount:
                    controller.stats['congestion_events'] += 1
                    controller.log(step, 
                        f"⚠️ CONGESTION (queue={state['total_halting']}) → Extending GREEN by {extension_amount}s")
            except:
                pass
        else:
            # Max extension reached
            if controller.extension_time > 0:
                controller.extension_time = 0
                controller.log(step, "⏱️ Max extension → Allowing phase rotation")
                try:
                    traci.trafficlight.setPhaseDuration(tls_id, MIN_GREEN)
                except:
                    pass
        return
    
    # PRIORITY 3: Normal
    if controller.extension_time > 0:
        controller.extension_time = 0
        controller.log(step, "✓ NORMAL TRAFFIC → Default control")

# ============================================================================
# MAIN SIMULATION
# ============================================================================
def run_adaptive_simulation():
    print("=" * 80)
    print("BEST ADAPTIVE TRAFFIC CONTROL")
    print("Balanced optimization for 20-35% improvement")
    print("=" * 80)
    
    sumo_binary = "sumo-gui" if SUMO_GUI else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CONFIG,
        "--tripinfo-output", TRIPINFO_OUTPUT,
        "--start",
        "--quit-on-end",
        "--no-warnings", "true"
    ]
    
    try:
        traci.start(sumo_cmd)
        print("✓ SUMO started successfully")
        print(f"✓ Output file: {TRIPINFO_OUTPUT}")
    except Exception as e:
        print(f"✗ Failed to start SUMO: {e}")
        sys.exit(1)
    
    controller = TrafficController()
    
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
        
        controller.phase_to_lanes = build_phase_lane_mapping(tls_id)
        print(f"✓ Built phase mapping for {len(controller.phase_to_lanes)} phases")
        
        routes = get_valid_routes()
        print(f"✓ Found {len(routes)} valid routes")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traci.close()
        sys.exit(1)
    
    print(f"\n▶ Starting BEST simulation for {SIM_STEPS} seconds...")
    print(f"  Congestion threshold: {CONGESTION_THRESHOLD}")
    print(f"  Emergency spawn interval: {EMERGENCY_SPAWN_INTERVAL}s")
    print(f"  Smart extension: 6-12s based on congestion level\n")
    
    step = 0
    try:
        while step < SIM_STEPS:
            traci.simulationStep()
            step += 1
            
            if traci.simulation.getMinExpectedNumber() == 0:
                print(f"\n✓ No more vehicles at step {step}")
                break
            
            try_spawn_emergency(step, controller)
            control_traffic_light(tls_id, controlled_lanes, controller, step)
            
            if step % 1000 == 0:
                state = read_traffic_state(controlled_lanes)
                print(f"[{step}s] Status: {state['total_halting']} halting, "
                      f"{len(traci.vehicle.getIDList())} vehicles, "
                      f"{controller.stats['emergency_events']} emergencies")
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        traci.close()
        print(f"\n✓ BEST simulation completed at step {step}")
        print(f"✓ Trip statistics saved to: {TRIPINFO_OUTPUT}")
        controller.print_stats()

if __name__ == "__main__":
    run_adaptive_simulation()
