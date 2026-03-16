
import traci
import sys
import random

# ============================================================================
# CONFIGURATION - MATCH ADAPTIVE
# ============================================================================
SUMO_CONFIG = "kpit.sumocfg"
SUMO_GUI = True
SIM_STEPS = 10000  # Match adaptive

# Output file
TRIPINFO_OUTPUT = "baseline_tripinfo.xml"

# Emergency vehicle spawning - SAME AS ADAPTIVE
EMERGENCY_TYPES = ["emergency", "ambulance", "fire", "police"]
EMERGENCY_SPAWN_INTERVAL = 100  # Same as adaptive
EMERGENCY_SPAWN_PROBABILITY = 0.5  # Same as adaptive

# ============================================================================
# STATE TRACKING
# ============================================================================
class BaselineState:
    def __init__(self):
        self.last_emergency_spawn = 0
        self.emergency_count = 0

# ============================================================================
# GET VALID ROUTES
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
# SPAWN EMERGENCY VEHICLES - SAME AS ADAPTIVE
# ============================================================================
def try_spawn_emergency(step, state):
    if step - state.last_emergency_spawn < EMERGENCY_SPAWN_INTERVAL:
        return
    
    if random.random() > EMERGENCY_SPAWN_PROBABILITY:
        state.last_emergency_spawn = step
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
        traci.vehicle.setMaxSpeed(vid, 35)  # Same speed as adaptive
        
        state.emergency_count += 1
        print(f"[{step}s] 🚨 Spawned {etype.upper()} → {vid} (NO priority - baseline)")
        state.last_emergency_spawn = step
    except:
        pass

# ============================================================================
# MAIN SIMULATION
# ============================================================================
def run_baseline_simulation():
    print("=" * 80)
    print("BASELINE TRAFFIC CONTROL - Fixed-Time Signals")
    print("For Fair Comparison with Ultra-Aggressive Adaptive")
    print("=" * 80)
    print("NOTE: Emergency vehicles spawn but get NO priority")
    print("      Fixed-time signals with default SUMO timing")
    print("=" * 80)
    
    sumo_binary = "sumo-gui" if SUMO_GUI else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CONFIG,
        "--tripinfo-output", TRIPINFO_OUTPUT,
        "--start",
        "--quit-on-end",
        "--no-warnings", "true",
        "--time-to-teleport", "600"  # Match adaptive
    ]
    
    try:
        traci.start(sumo_cmd)
        print(f"✓ SUMO started successfully")
        print(f"✓ Using FIXED-TIME traffic signals (default SUMO)")
        print(f"✓ Output file: {TRIPINFO_OUTPUT}")
    except Exception as e:
        print(f"✗ Failed to start SUMO: {e}")
        sys.exit(1)
    
    state = BaselineState()
    
    try:
        tls_list = traci.trafficlight.getIDList()
        if tls_list:
            print(f"✓ Traffic lights: {len(tls_list)} junction(s) (fixed-time)")
        
        routes = get_valid_routes()
        print(f"✓ Found {len(routes)} valid routes")
    except:
        pass
    
    print(f"\n▶ Starting BASELINE simulation for {SIM_STEPS} seconds...")
    print(f"  Settings:")
    print(f"    • Fixed-time signals (NO adaptive control)")
    print(f"    • Emergency vehicles spawn (NO priority)")
    print(f"    • Emergency interval: {EMERGENCY_SPAWN_INTERVAL}s (same as adaptive)\n")
    
    step = 0
    try:
        while step < SIM_STEPS:
            traci.simulationStep()
            step += 1
            
            if traci.simulation.getMinExpectedNumber() == 0:
                print(f"\n✓ No more vehicles at step {step}")
                break
            
            # Spawn emergency vehicles (SAME AS ADAPTIVE)
            try_spawn_emergency(step, state)
            
            # NO TRAFFIC CONTROL - Let SUMO use default fixed-time signals
            
            if step % 1000 == 0:
                num_vehicles = len(traci.vehicle.getIDList())
                print(f"[{step}s] Status: {num_vehicles} vehicles, "
                      f"{state.emergency_count} emergencies spawned")
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        traci.close()
        print(f"\n✓ BASELINE simulation completed at step {step}")
        print(f"✓ Total emergency vehicles spawned: {state.emergency_count}")
        print(f"✓ Trip statistics saved to: {TRIPINFO_OUTPUT}")
        print("=" * 80)

if __name__ == "__main__":
    run_baseline_simulation()
