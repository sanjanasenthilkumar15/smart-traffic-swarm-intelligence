"""
MQTT DASHBOARD - Real-Time Traffic Monitoring

This dashboard subscribes to all traffic MQTT topics and displays
real-time updates in a clean console interface.
"""

import paho.mqtt.client as mqtt
import json
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# ============================================================================
# STATISTICS
# ============================================================================
class Stats:
    def __init__(self):
        self.messages_received = 0
        self.state_updates = 0
        self.emergency_alerts = 0
        self.control_commands = 0
        self.health_updates = 0
        self.junctions = set()

stats = Stats()

# ============================================================================
# MESSAGE HANDLERS
# ============================================================================
def format_timestamp():
    """Get formatted timestamp"""
    return datetime.now().strftime('%H:%M:%S')

def handle_state_message(topic, data):
    """Handle traffic state update"""
    stats.state_updates += 1
    stats.junctions.add(data.get('junction_id', 'Unknown'))
    
    print(f"\n{'='*70}")
    print(f"🚦 TRAFFIC STATE UPDATE [{format_timestamp()}]")
    print(f"{'='*70}")
    print(f"Junction: {data.get('junction_id')}")
    print(f"Current Phase: {data.get('current_phase')}")
    print(f"Phase Duration: {data.get('phase_duration', 0):.1f}s")
    print(f"Queue Lengths: {data.get('queues')}")
    print(f"Average Speeds: {data.get('speeds')} m/s")
    print(f"Total Vehicles: {data.get('total_vehicles', 0)}")
    
    if data.get('emergency_flag'):
        print(f"⚠️  EMERGENCY VEHICLE PRESENT!")
    else:
        print(f"✓ Normal Traffic")
    
    print(f"{'='*70}")

def handle_emergency_message(topic, data):
    """Handle emergency alert"""
    stats.emergency_alerts += 1
    
    print(f"\n{'#'*70}")
    print(f"🚨🚨 EMERGENCY ALERT 🚨🚨 [{format_timestamp()}]")
    print(f"{'#'*70}")
    print(f"Junction: {data.get('junction_id')}")
    
    vehicles = data.get('vehicles', [])
    print(f"Emergency Vehicles: {len(vehicles)}")
    
    for i, veh in enumerate(vehicles[:3], 1):  # Show first 3
        print(f"  {i}. Type: {veh.get('type', 'Unknown').upper()}")
        print(f"     ID: {veh.get('id', 'Unknown')}")
    
    print(f"Current Phase: {data.get('current_phase')}")
    print(f"ETA: {data.get('eta', 0)}s")
    print(f"{'#'*70}")

def handle_control_message(topic, data):
    """Handle control command"""
    stats.control_commands += 1
    
    print(f"\n{'─'*70}")
    print(f"📡 CONTROL COMMAND [{format_timestamp()}]")
    print(f"{'─'*70}")
    print(f"Command: {data.get('command')}")
    print(f"Target Phase: {data.get('phase')}")
    print(f"Priority: {data.get('priority', 'NORMAL')}")
    print(f"{'─'*70}")

def handle_health_message(topic, data):
    """Handle health status"""
    stats.health_updates += 1
    
    print(f"\n{'─'*70}")
    print(f"💚 SYSTEM HEALTH [{format_timestamp()}]")
    print(f"{'─'*70}")
    print(f"Junction: {data.get('junction_id')}")
    print(f"MQTT Connected: {'✓ Yes' if data.get('mqtt_connected') else '✗ No'}")
    print(f"Messages Published: {data.get('messages_published', 0)}")
    print(f"Messages Received: {data.get('messages_received', 0)}")
    print(f"AI Inferences: {data.get('ai_inferences', 0)}")
    print(f"Phase Changes: {data.get('phase_changes', 0)}")
    print(f"Emergencies Served: {data.get('emergencies_served', 0)}")
    print(f"Uptime: {data.get('uptime_seconds', 0)}s")
    print(f"{'─'*70}")

def handle_sync_message(topic, data):
    """Handle synchronization command"""
    print(f"\n{'─'*70}")
    print(f"🔄 SYNC COMMAND [{format_timestamp()}]")
    print(f"{'─'*70}")
    print(f"Corridor: {data.get('corridor', [])}")
    print(f"Vehicle: {data.get('vehicle', 'Unknown')}")
    print(f"{'─'*70}")

# ============================================================================
# MQTT CALLBACKS
# ============================================================================
def on_connect(client, userdata, flags, rc):
    """Callback when connected to broker"""
    if rc == 0:
        print(f"\n✓ Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        print(f"✓ Subscribing to traffic/#")
        client.subscribe("traffic/#")
        print(f"\n{'='*70}")
        print(f"MQTT DASHBOARD RUNNING")
        print(f"{'='*70}")
        print(f"Monitoring all traffic topics...")
        print(f"Press Ctrl+C to exit\n")
    else:
        print(f"✗ Connection failed with code {rc}")
        sys.exit(1)

def on_message(client, userdata, msg):
    """Callback when message received"""
    try:
        stats.messages_received += 1
        
        # Parse JSON payload
        try:
            data = json.loads(msg.payload.decode())
        except:
            # Not JSON, show raw
            print(f"\n📨 Raw Message: {msg.topic}")
            print(f"   {msg.payload.decode()}")
            return
        
        # Route to appropriate handler
        topic = msg.topic
        
        if '/state' in topic:
            handle_state_message(topic, data)
            
        elif '/emergency' in topic:
            handle_emergency_message(topic, data)
            
        elif '/control' in topic:
            handle_control_message(topic, data)
            
        elif '/health' in topic:
            handle_health_message(topic, data)
            
        elif '/sync' in topic:
            handle_sync_message(topic, data)
            
        else:
            # Unknown topic
            print(f"\n📨 Unknown Topic: {topic}")
            print(f"   Data: {data}")
    
    except Exception as e:
        print(f"\n✗ Error processing message: {e}")
        print(f"   Topic: {msg.topic}")
        print(f"   Payload: {msg.payload}")

def on_disconnect(client, userdata, rc):
    """Callback when disconnected"""
    print(f"\n⚠ Disconnected from broker (code {rc})")
    if rc != 0:
        print(f"Unexpected disconnection. Attempting to reconnect...")

# ============================================================================
# STATISTICS DISPLAY
# ============================================================================
def print_stats():
    """Print dashboard statistics"""
    print(f"\n{'='*70}")
    print(f"DASHBOARD STATISTICS")
    print(f"{'='*70}")
    print(f"Total Messages: {stats.messages_received}")
    print(f"State Updates: {stats.state_updates}")
    print(f"Emergency Alerts: {stats.emergency_alerts}")
    print(f"Control Commands: {stats.control_commands}")
    print(f"Health Updates: {stats.health_updates}")
    print(f"Active Junctions: {len(stats.junctions)}")
    if stats.junctions:
        print(f"Junctions: {', '.join(sorted(stats.junctions))}")
    print(f"{'='*70}\n")

# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main dashboard loop"""
    
    print(f"\n{'='*70}")
    print(f"MQTT TRAFFIC MONITORING DASHBOARD")
    print(f"{'='*70}")
    print(f"Connecting to MQTT broker...")
    print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}\n")
    
    # Create MQTT client
    client = mqtt.Client(client_id="traffic_dashboard")
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    
    # Connect to broker
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"✗ Failed to connect to MQTT broker: {e}")
        print(f"\nMake sure the MQTT broker is running:")
        print(f"  mosquitto -v")
        sys.exit(1)
    
    # Start listening
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print(f"\n\n⚠ Dashboard stopped by user")
        print_stats()
        client.disconnect()

if __name__ == "__main__":
    main()
