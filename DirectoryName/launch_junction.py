"""
Junction Launcher - Easy way to start junctions
"""
import os
import sys
import subprocess

print("=" * 55)
print("   Multi-Junction Traffic Control Launcher")
print("=" * 55)
print("")
print("Select junction to run:")
print("  1. Junction J1")
print("  2. Junction J2")
print("  3. Junction J3")
print("  4. Exit")
print("")

choice = input("Enter choice (1-4): ")

junction_map = {
    '1': 'J1',
    '2': 'J2',
    '3': 'J3'
}

if choice in junction_map:
    junction_id = junction_map[choice]
    
    print(f"\nStarting Junction {junction_id}...")
    print("Press Ctrl+C to stop\n")
    
    # Set environment variables
    os.environ['JUNCTION_ID'] = junction_id
    os.environ['SUMO_CONFIG'] = 'kpit.sumocfg'
    
    # Run the simulation
    subprocess.run(['python', 'adaptive_signal_mqtt.py'])
    
elif choice == '4':
    print("Exiting...")
    sys.exit(0)
else:
    print("Invalid choice! Please enter 1, 2, 3 or 4.")