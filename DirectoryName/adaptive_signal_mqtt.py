"""
PHASE 3: MQTT-ENABLED EDGE AI TRAFFIC CONTROLLER

This version adds:
- MQTT publishing of traffic state
- MQTT subscription for control commands
- Multi-junction synchronization
- Emergency vehicle corridor system
- Cloud connectivity
"""

import traci
import sys
import random
import numpy as np
import pickle
import json
import time
from typing import List, Dict, Tuple
from datetime import datetime

# ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# MQTT Client
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("⚠ MQTT not available. Install with: pip install paho-mqtt")

# ============================================================================
# CONFIGURATION
# ============================================================================
SUMO_CONFIG = "kpit.sumocfg"
SUMO_GUI = True
SIM_STEPS = 10000

# AI Model files
ONNX_MODEL = "traffic_model.onnx"
SCALER_FILE = "traffic_scaler.pkl"
CONFIG_FILE = "model_config.pkl"

#Output files
TRIPINFO_OUTPUT = "adaptive_tripinfo.xml"


# MQTT Configuration
MQTT_BROKER = "localhost"  # Change to your broker address
MQTT_PORT = 1883
MQTT_KEEPALIVE = 60
CITY_NAME = "kpit"
JUNCTION_ID = "J1"  # Your junction identifier

# MQTT Topics
TOPIC_STATE = f"traffic/{CITY_NAME}/{JUNCTION_ID}/state"
TOPIC_CONTROL = f"traffic/{CITY_NAME}/{JUNCTION_ID}/control"
TOPIC_EMERGENCY_LOCAL = f"traffic/{CITY_NAME}/{JUNCTION_ID}/emergency"
TOPIC_EMERGENCY_GLOBAL = f"traffic/{CITY_NAME}/emergency/alerts"
TOPIC_SYNC = f"traffic/{CITY_NAME}/global/sync"
TOPIC_HEALTH = f"traffic/{CITY_NAME}/{JUNCTION_ID}/health"

# Safety constraints
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 60
EMERGENCY_GREEN_TIME = 45

# Emergency vehicles
EMERGENCY_VEHICLES = {
    "ambulance": {"color": (255, 255, 255, 255), "icon": "🚑", "priority": 1, "max_speed": 30},
    "fire_truck": {"color": (255, 0, 0, 255), "icon": "🚒", "priority": 1, "max_speed": 28},
    "police": {"color": (0, 0, 255, 255), "icon": "🚓", "priority": 2, "max_speed": 32},
    "rescue": {"color": (255, 165, 0, 255), "icon": "🚐", "priority": 2, "max_speed": 26},
    "paramedic": {"color": (0, 255, 0, 255), "icon": "🚙", "priority": 1, "max_speed": 29},
}

EMERGENCY_SPAWN_INTERVAL = 100
EMERGENCY_SPAWN_PROBABILITY = 0.5
MULTI_SPAWN_PROBABILITY = 0.3
MAX_SIMULTANEOUS_SPAWN = 2

# ============================================================================
# MQTT CLIENT MANAGER
# ============================================================================
class MQTTManager:
    """Manages MQTT connections and message handling"""
    
    def __init__(self, broker, port, junction_id):
        self.broker = broker
        self.port = port
        self.junction_id = junction_id
        self.client = None
        self.connected = False
        self.messages_published = 0
        self.messages_received = 0
        self.remote_commands = []
        
    def initialize(self):
        """Initialize MQTT client"""
        if not MQTT_AVAILABLE:
            print("✗ MQTT not available")
            return False
        
        try:
            self.client = mqtt.Client(client_id=f"edge_{self.junction_id}")
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            
            print(f"\n{'='*80}")
            print("MQTT CLIENT INITIALIZATION")
            print(f"{'='*80}")
            print(f"Broker: {self.broker}:{self.port}")
            print(f"Junction ID: {self.junction_id}")
            
            self.client.connect(self.broker, self.port, MQTT_KEEPALIVE)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 5
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.connected:
                print(f"✓ Connected to MQTT broker")
                print(f"{'='*80}\n")
                return True
            else:
                print(f"✗ Failed to connect to MQTT broker (timeout)")
                return False
                
        except Exception as e:
            print(f"✗ MQTT initialization failed: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to broker"""
        if rc == 0:
            self.connected = True
            print(f"✓ MQTT connected with result code {rc}")
            
            # Subscribe to control topics
            subscriptions = [
                (TOPIC_CONTROL, 0),
                (TOPIC_EMERGENCY_GLOBAL, 0),
                (TOPIC_SYNC, 0)
            ]
            
            for topic, qos in subscriptions:
                self.client.subscribe(topic, qos)
                print(f"  📡 Subscribed to: {topic}")
        else:
            print(f"✗ MQTT connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from broker"""
        self.connected = False
        print(f"⚠ MQTT disconnected with result code {rc}")
    
    def _on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            payload = json.loads(msg.payload.decode())
            self.messages_received += 1
            
            print(f"\n📥 MQTT Message Received")
            print(f"   Topic: {msg.topic}")
            print(f"   Payload: {payload}")
            
            # Store command for processing
            self.remote_commands.append({
                'topic': msg.topic,
                'payload': payload,
                'timestamp': time.time()
            })
            
        except Exception as e:
            print(f"✗ Error processing MQTT message: {e}")
    
    def publish_state(self, state_data):
        """Publish traffic state to MQTT"""
        if not self.connected:
            return False
        
        try:
            payload = json.dumps(state_data)
            result = self.client.publish(TOPIC_STATE, payload, qos=0)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.messages_published += 1
                return True
            return False
            
        except Exception as e:
            print(f"✗ Error publishing state: {e}")
            return False
    
    def publish_emergency(self, emergency_data):
        """Publish emergency vehicle alert"""
        if not self.connected:
            return False
        
        try:
            payload = json.dumps(emergency_data)
            
            # Publish to both local and global emergency topics
            self.client.publish(TOPIC_EMERGENCY_LOCAL, payload, qos=1)
            self.client.publish(TOPIC_EMERGENCY_GLOBAL, payload, qos=1)
            
            print(f"\n🚨 EMERGENCY ALERT PUBLISHED")
            print(f"   {emergency_data}")
            
            self.messages_published += 2
            return True
            
        except Exception as e:
            print(f"✗ Error publishing emergency: {e}")
            return False
    
    def publish_health(self, health_data):
        """Publish system health status"""
        if not self.connected:
            return False
        
        try:
            payload = json.dumps(health_data)
            self.client.publish(TOPIC_HEALTH, payload, qos=0)
            return True
        except:
            return False
    
    def get_pending_commands(self):
        """Get and clear pending remote commands"""
        commands = self.remote_commands.copy()
        self.remote_commands.clear()
        return commands
    
    def shutdown(self):
        """Shutdown MQTT client"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            print(f"\n✓ MQTT client shutdown")
            print(f"  Messages published: {self.messages_published}")
            print(f"  Messages received: {self.messages_received}")

# ============================================================================
# AI INFERENCE ENGINE
# ============================================================================
class EdgeAIController:
    """Edge AI inference engine"""
    
    def __init__(self):
        self.onnx_session = None
        self.scaler = None
        self.config = None
        self.fallback_mode = False
        self.inference_count = 0
        
    def initialize(self):
        """Load ONNX model"""
        
        print(f"\n{'='*80}")
        print("EDGE AI CONTROLLER INITIALIZATION")
        print(f"{'='*80}")
        
        if not ONNX_AVAILABLE:
            print("✗ ONNX Runtime not available")
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
            print(f"✓ ONNX model loaded: {ONNX_MODEL}")
            print(f"  Ready for edge inference")
        except Exception as e:
            print(f"✗ Failed to load ONNX model: {e}")
            self.fallback_mode = True
            return False
        
        print(f"{'='*80}\n")
        return True
    
    def predict_phase(self, traffic_state: Dict) -> int:
        """AI inference"""
        
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
            return self._rule_based_fallback(traffic_state)
    
    def _rule_based_fallback(self, state: Dict) -> int:
        """Fallback control"""
        current_phase = state['current_phase']
        total_queue = sum(state['queues'])
        
        if state['emergency_flag'] == 1:
            return current_phase
        
        if total_queue >= 10:
            num_phases = self.config['num_phases'] if self.config else 3
            return (current_phase + 1) % num_phases
        
        return current_phase

# ============================================================================
# SYSTEM STATE
# ============================================================================
class TrafficSystemState:
    """System state manager"""
    
    def __init__(self):
        self.emergency_mode = False
        self.emergency_start_time = 0
        self.last_emergency_spawn = 0
        self.last_phase_change = 0
        self.current_phase_duration = 0
        self.phase_changes = 0
        self.emergency_vehicles_served = 0
        self.emergency_by_type = {}
        self.mqtt_override_active = False
        self.last_state_publish = 0
        self.last_health_publish = 0

# ============================================================================
# TRAFFIC STATE EXTRACTION
# ============================================================================
def extract_traffic_state(tls_id: str, controlled_lanes: List[str]) -> Dict:
    """Extract traffic state"""
    
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
    
    # Emergency detection
    emergency_flag = 0
    emergency_vehicles = []
    emergency_details = []
    
    try:
        all_vehicles = traci.vehicle.getIDList()
        for veh_id in all_vehicles:
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
                                'lane': veh_lane,
                                'priority': EMERGENCY_VEHICLES[veh_type]['priority']
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
# HELPER FUNCTIONS
# ============================================================================
def get_num_phases(tls_id: str) -> int:
    """Get number of phases"""
    try:
        return traci.trafficlight.getPhaseNumber(tls_id)
    except AttributeError:
        try:
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            return len(logic.phases)
        except:
            return 3

def apply_safety_constraints(predicted_phase: int, current_phase: int, 
                            time_in_phase: float, emergency_mode: bool, 
                            num_phases: int) -> Tuple[int, bool]:
    """Safety validation"""
    
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

def execute_phase_change(tls_id: str, target_phase: int, state: TrafficSystemState):
    """Execute phase change"""
    try:
        current_phase = traci.trafficlight.getPhase(tls_id)
        
        if target_phase != current_phase:
            traci.trafficlight.setPhase(tls_id, target_phase)
            state.phase_changes += 1
            state.last_phase_change = traci.simulation.getTime()
            state.current_phase_duration = 0
            return True
        return False
    except:
        return False

def get_valid_routes() -> List[str]:
    """Get valid routes"""
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
    """Spawn emergency vehicles"""
    
    if step - state.last_emergency_spawn < EMERGENCY_SPAWN_INTERVAL:
        return
    
    if random.random() > EMERGENCY_SPAWN_PROBABILITY:
        state.last_emergency_spawn = step
        return
    
    routes = get_valid_routes()
    if not routes:
        return
    
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
            
            veh_config = EMERGENCY_VEHICLES[veh_type]
            traci.vehicle.setColor(vid, veh_config['color'])
            traci.vehicle.setMaxSpeed(vid, veh_config['max_speed'])
            traci.vehicle.setLength(vid, 6.0)
            
            spawned.append({
                'id': vid,
                'type': veh_type,
                'icon': veh_config['icon']
            })
            
            if veh_type not in state.emergency_by_type:
                state.emergency_by_type[veh_type] = 0
            state.emergency_by_type[veh_type] += 1
            
        except:
            pass
    
    if spawned:
        state.last_emergency_spawn = step
        print(f"\n{'='*60}")
        print(f"[{step}s] 🚨 EMERGENCY VEHICLES SPAWNED!")
        for vehicle in spawned:
            print(f"  {vehicle['icon']} {vehicle['type'].upper().replace('_', ' ')}")
        print(f"{'='*60}\n")

def get_controlled_lanes(tls_id: str) -> List[str]:
    """Get controlled lanes"""
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
# MQTT MESSAGE HANDLERS
# ============================================================================
def process_remote_commands(commands, tls_id, system_state, num_phases):
    """Process commands received via MQTT"""
    
    for cmd in commands:
        topic = cmd['topic']
        payload = cmd['payload']
        
        # Control command
        if 'control' in topic:
            if payload.get('command') == 'SET_PHASE':
                target_phase = payload.get('phase', 0)
                priority = payload.get('priority', 'NORMAL')
                
                if 0 <= target_phase < num_phases:
                    if priority == 'EMERGENCY' or not system_state.emergency_mode:
                        try:
                            traci.trafficlight.setPhase(tls_id, target_phase)
                            system_state.mqtt_override_active = True
                            print(f"\n📡 MQTT CONTROL: Phase set to {target_phase} (Priority: {priority})")
                        except:
                            pass
        
        # Emergency alert from other junction
        elif 'emergency' in topic and 'alerts' in topic:
            vehicle_id = payload.get('vehicle_id', '')
            eta = payload.get('eta', 0)
            
            print(f"\n🚨 EMERGENCY ALERT FROM NETWORK:")
            print(f"   Vehicle: {vehicle_id}")
            print(f"   ETA: {eta}s")
            print(f"   Preparing for corridor...")
        
        # Global sync command
        elif 'sync' in topic:
            corridor = payload.get('corridor', [])
            if JUNCTION_ID in corridor:
                print(f"\n🔄 SYNC COMMAND: Joining corridor {corridor}")

# ============================================================================
# MAIN SIMULATION
# ============================================================================
def run_mqtt_enabled_simulation():
    """Main simulation with MQTT integration"""
    
    print("\n" + "=" * 80)
    print("🌐 PHASE 3: MQTT-ENABLED EDGE AI TRAFFIC CONTROL 🌐")
    print("=" * 80)
    print("Features:")
    print("  • MQTT State Publishing")
    print("  • Remote Control via MQTT")
    print("  • Emergency Vehicle Corridors")
    print("  • Multi-Junction Coordination")
    print("  • Cloud Connectivity")
    print("=" * 80)
    
    # Initialize MQTT
    mqtt_manager = MQTTManager(MQTT_BROKER, MQTT_PORT, JUNCTION_ID)
    mqtt_enabled = mqtt_manager.initialize()
    
    if not mqtt_enabled:
        print("\n⚠ Running without MQTT (standalone mode)")
    
    # Initialize AI
    ai_controller = EdgeAIController()
    ai_controller.initialize()
    
    system_state = TrafficSystemState()
    
    # Start SUMO
    print(f"\n{'='*80}")
    print("STARTING SUMO SIMULATION")
    print(f"{'='*80}")
    
    sumo_binary = "sumo-gui" if SUMO_GUI else "sumo"
    sumo_cmd = [sumo_binary, "-c", SUMO_CONFIG, "--start", "--quit-on-end", "--tripinfo-output", TRIPINFO_OUTPUT]
    
    try:
        traci.start(sumo_cmd)
        print("✓ SUMO started")
    except Exception as e:
        print(f"✗ SUMO failed: {e}")
        if mqtt_enabled:
            mqtt_manager.shutdown()
        sys.exit(1)
    
    # Get traffic light info
    try:
        tls_list = traci.trafficlight.getIDList()
        if not tls_list:
            print("✗ No traffic lights")
            traci.close()
            if mqtt_enabled:
                mqtt_manager.shutdown()
            sys.exit(1)
        
        tls_id = tls_list[0]
        print(f"✓ Traffic light: {tls_id}")
        
        controlled_lanes = get_controlled_lanes(tls_id)
        print(f"✓ Lanes: {len(controlled_lanes)}")
        
        num_phases = get_num_phases(tls_id)
        print(f"✓ Phases: {num_phases}")
        
    except Exception as e:
        print(f"✗ Init error: {e}")
        traci.close()
        if mqtt_enabled:
            mqtt_manager.shutdown()
        sys.exit(1)
    
    print(f"\n▶ Starting MQTT-enabled simulation...")
    print(f"  Junction ID: {JUNCTION_ID}")
    print(f"  MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"{'='*80}\n")
    
    # Main loop
    step = 0
    last_status_print = 0
    STATE_PUBLISH_INTERVAL = 1  # Publish every second
    HEALTH_PUBLISH_INTERVAL = 10  # Publish health every 10 seconds
    
    try:
        while step < SIM_STEPS:
            traci.simulationStep()
            step += 1
            current_time = traci.simulation.getTime()
            
            if traci.simulation.getMinExpectedNumber() == 0:
                break
            
            # Spawn emergency vehicles
            try_spawn_emergency(step, system_state)
            
            # Extract traffic state
            traffic_state = extract_traffic_state(tls_id, controlled_lanes)
            
            # MQTT: Publish traffic state
            if mqtt_enabled and (step - system_state.last_state_publish) >= STATE_PUBLISH_INTERVAL:
                state_data = {
                    "junction_id": JUNCTION_ID,
                    "timestamp": datetime.now().isoformat(),
                    "simulation_time": current_time,
                    "queues": traffic_state['queues'],
                    "speeds": traffic_state['speeds'],
                    "emergency_flag": traffic_state['emergency_flag'],
                    "current_phase": traffic_state['current_phase'],
                    "phase_duration": system_state.current_phase_duration,
                    "total_vehicles": len(traci.vehicle.getIDList())
                }
                
                mqtt_manager.publish_state(state_data)
                system_state.last_state_publish = step
            
            # MQTT: Process remote commands
            if mqtt_enabled:
                commands = mqtt_manager.get_pending_commands()
                if commands:
                    process_remote_commands(commands, tls_id, system_state, num_phases)
            
            # Emergency handling
            if traffic_state['emergency_flag'] == 1:
                if not system_state.emergency_mode:
                    system_state.emergency_mode = True
                    system_state.emergency_start_time = current_time
                    system_state.emergency_vehicles_served += 1
                    
                    print(f"\n{'#'*60}")
                    print(f"[{step}s] 🚨 EMERGENCY MODE")
                    print(f"{'#'*60}")
                    
                    for detail in traffic_state['emergency_details']:
                        print(f"  {detail['icon']} {detail['type'].upper()}: {detail['id']}")
                    
                    print(f"  🚦 HOLDING GREEN")
                    print(f"{'#'*60}\n")
                    
                    # MQTT: Publish emergency alert
                    if mqtt_enabled:
                        emergency_data = {
                            "junction_id": JUNCTION_ID,
                            "timestamp": datetime.now().isoformat(),
                            "vehicles": traffic_state['emergency_details'],
                            "current_phase": traffic_state['current_phase'],
                            "eta": EMERGENCY_GREEN_TIME
                        }
                        mqtt_manager.publish_emergency(emergency_data)
                
                continue
            
            # Clear emergency
            if system_state.emergency_mode and traffic_state['emergency_flag'] == 0:
                system_state.emergency_mode = False
                system_state.mqtt_override_active = False
                print(f"\n[{step}s] ✓ EMERGENCY CLEARED\n")
            
            # AI inference (if not overridden by MQTT)
            if not system_state.mqtt_override_active:
                predicted_phase = ai_controller.predict_phase(traffic_state)
                
                time_in_phase = current_time - system_state.last_phase_change
                system_state.current_phase_duration = time_in_phase
                
                approved_phase, can_change = apply_safety_constraints(
                    predicted_phase, traffic_state['current_phase'],
                    time_in_phase, system_state.emergency_mode, num_phases
                )
                
                if can_change and approved_phase != traffic_state['current_phase']:
                    changed = execute_phase_change(tls_id, approved_phase, system_state)
                    if changed:
                        print(f"[{step}s] 🤖 AI: Phase {traffic_state['current_phase']} → {approved_phase}")
            
            # MQTT: Publish health status
            if mqtt_enabled and (step - system_state.last_health_publish) >= HEALTH_PUBLISH_INTERVAL:
                health_data = {
                    "junction_id": JUNCTION_ID,
                    "timestamp": datetime.now().isoformat(),
                    "mqtt_connected": mqtt_manager.connected,
                    "messages_published": mqtt_manager.messages_published,
                    "messages_received": mqtt_manager.messages_received,
                    "ai_inferences": ai_controller.inference_count,
                    "phase_changes": system_state.phase_changes,
                    "emergencies_served": system_state.emergency_vehicles_served,
                    "uptime_seconds": step
                }
                mqtt_manager.publish_health(health_data)
                system_state.last_health_publish = step
            
            # Status updates
            if step - last_status_print >= 500:
                num_vehicles = len(traci.vehicle.getIDList())
                
                print(f"\n{'─'*60}")
                print(f"[{step}s] 📊 STATUS")
                print(f"{'─'*60}")
                print(f"  🚗 Vehicles: {num_vehicles}")
                print(f"  🚦 Phase Changes: {system_state.phase_changes}")
                print(f"  🚨 Emergencies: {system_state.emergency_vehicles_served}")
                
                if mqtt_enabled:
                    print(f"  📡 MQTT Published: {mqtt_manager.messages_published}")
                    print(f"  📥 MQTT Received: {mqtt_manager.messages_received}")
                
                print(f"{'─'*60}\n")
                last_status_print = step
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        traci.close()
        
        if mqtt_enabled:
            mqtt_manager.shutdown()
        
        # Summary
        print(f"\n{'='*80}")
        print("SIMULATION SUMMARY")
        print(f"{'='*80}")
        print(f"Duration: {step}s")
        print(f"Phase Changes: {system_state.phase_changes}")
        print(f"Emergencies: {system_state.emergency_vehicles_served}")
        
        if mqtt_enabled:
            print(f"\nMQTT Statistics:")
            print(f"  Published: {mqtt_manager.messages_published}")
            print(f"  Received: {mqtt_manager.messages_received}")
        
        print(f"{'='*80}\n")

if __name__ == "__main__":
    run_mqtt_enabled_simulation()
