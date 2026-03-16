# Swarm Intelligence-Based Smart Traffic Management for Indian Cities

> Adaptive, decentralized traffic signal control system — evolved from rule-based logic to a neural network ML model with MQTT multi-agent coordination, simulated and validated in SUMO.

[![KPIT Sparkle](https://img.shields.io/badge/KPIT_Sparkle-National_Finalist-brightgreen?style=flat)](https://sparkle.kpit.com)
[![Demo](https://img.shields.io/badge/Demo-Watch_Video-red?style=flat&logo=youtube)](YOUR_DEMO_LINK)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)]()
[![SUMO](https://img.shields.io/badge/SUMO-Simulation-blue?style=flat)]()
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-gray?style=flat)]()
[![MQTT](https://img.shields.io/badge/MQTT-Protocol-purple?style=flat)]()

---

## The Problem

Indian city traffic signals run on fixed timers — green for 30 seconds whether 50 cars are waiting or none. They cannot adapt to real-time congestion, emergencies, or unpredictable Indian traffic conditions. The result: unnecessary delays, fuel waste, and blocked emergency corridors.

---

## Our Solution

A **swarm intelligence-based adaptive traffic controller** where each intersection acts as an intelligent, self-organizing agent. Intersections coordinate locally with neighbours using MQTT to optimize signal timings in real time — tested entirely in a SUMO simulation with a trained neural network model.

**Team InnovateX · RMK Engineering College · KPIT Sparkle Techathon — National Finalist**
**Category:** Virtual Engineering and Simulation Driven Systems for Intelligent Vehicle Engineering

---

## Development Journey

This project was built in three distinct phases — each improving on the last:

### Phase 1 — Rule-based baseline (`baseline_system.py`)
Started with a fixed rule-based signal controller as the comparison baseline.
Signals switch on hard-coded thresholds — no learning, no adaptation.
This is the system we beat.

### Phase 2 — Neural network ML model (`train_model.py` → `traffic_model.onnx`)
Replaced the rule logic with a trained ML model that predicts optimal signal timings
based on real-time queue length and vehicle density.
- Trained with `train_model.py`, exported to `traffic_model.onnx`
- Runs inference via ONNX Runtime (`traffic_model.onnx`) — lightweight, edge-deployable
- Scaler saved as `traffic_scaler.pkl` for consistent input normalization
- Training history tracked in `training_history.png`

### Phase 3 — MQTT multi-agent coordination (`adaptive_signal_mqtt.py` + `mqtt_dashboard.py`)
Added MQTT protocol so intersections publish and subscribe to each other's states —
enabling true swarm coordination across multiple junctions.
- `adaptive_signal_mqtt.py` — each junction publishes its signal state
- `mqtt_dashboard.py` — real-time dashboard aggregating all junction states
- `adaptive_system.py` — full adaptive controller combining ML + MQTT
- Coordination latency: 3.2s swarm consensus across intersections

---

## Simulation Results

Swarm-AI vs Fixed-Timer baseline across 2–4 intersections in SUMO:

| Metric           | Fixed Baseline | Swarm-AI     | Improvement     |
|------------------|---------------|--------------|-----------------|
| Avg Delay        | 98.7 sec      | 87.6 sec     | **11% faster**  |
| Max Queue        | 124 vehicles  | 116 vehicles | **6% shorter**  |
| Throughput       | 118 veh/hr    | 200 veh/hr   | **69% increase**|
| Vehicles Cleared | 10 stalled    | 0 stalled    | **70% better**  |
| Edge AI Response | —             | <20ms         | Real-time       |

Full comparison data: `comparison_results.csv` · Traffic training data: `traffic_training_data.csv`

---

## Key Features

- **Swarm-Intelligent Coordination** — intersections act as self-organizing agents via MQTT
- **Neural Network Signal Control** — ML model replaces rigid rule logic, improves over time
- **ONNX Edge Deployment** — lightweight inference, runs on Raspberry Pi / NUC
- **Emergency Vehicle Priority** — auto green corridor for ambulances and buses
- **Before/After Comparison** — baseline vs adaptive, visualized in real time
- **Multi-Intersection Support** — 2–4 junctions coordinated simultaneously

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Traffic Simulation | SUMO (`kpit.sumocfg`, `road.net.net.xml`, `tripinfos.xml`) |
| ML Model | Python, PyTorch → ONNX Runtime (`traffic_model.onnx`) |
| Swarm Coordination | MQTT protocol (`adaptive_signal_mqtt.py`) |
| RL Algorithms | Q-learning, PPO, multi-agent reinforcement learning |
| Edge Inference | ONNX Runtime, `traffic_scaler.pkl` |
| Dashboard | `mqtt_dashboard.py` — real-time signal state monitoring |
| Data | `traffic_training_data.csv`, `comparison_results.csv` |

---

## Project Structure

```
DirectoryName/
├── baseline_system.py        # Rule-based fixed controller (baseline)
├── adaptive_signal.py        # Core adaptive signal controller
├── adaptive_signal_mqtt.py   # MQTT-enabled multi-agent controller
├── adaptive_system.py        # Full system combining ML + MQTT
├── mqtt_dashboard.py         # Real-time MQTT dashboard
├── train_model.py            # Neural network training script
├── export_to_onnx.py         # Export trained model to ONNX
├── data_logger.py            # Simulation data logging
├── launch_junction.py        # Junction launcher
├── comparison.py             # Baseline vs adaptive comparison
├── app.py                    # Main application entry point
│
├── traffic_model.onnx        # Trained ONNX model (edge-ready)
├── traffic_model.pth         # PyTorch model weights
├── traffic_scaler.pkl        # Input normalization scaler
├── model_config.pkl          # Model configuration
│
├── kpit.sumocfg              # SUMO simulation config
├── kpit.net.xml              # SUMO network definition
├── road.net.net.xml          # Road network
├── adaptive_tripinfo.xml     # Adaptive simulation trip data
├── baseline_tripinfo.xml     # Baseline simulation trip data
├── tripinfos.xml             # Combined trip information
│
├── traffic_training_data.csv # Training dataset
├── comparison_results.csv    # Baseline vs swarm-AI results
├── training_history.png      # Model training loss/accuracy curves
│
├── run_junction_J1.bat       # Run junction 1 simulation
├── run_junction_J2.bat       # Run junction 2 simulation
├── run_junction_J3.bat       # Run junction 3 simulation
│
└── requirements.txt          # Python dependencies
```

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
# Also install: SUMO (sumo-gui), Python 3.8+
```

### Run the baseline (fixed-timer)
```bash
python baseline_system.py
```

### Train the neural network model
```bash
python train_model.py
python export_to_onnx.py
```

### Run the adaptive swarm system
```bash
# Single junction
python adaptive_signal.py

# Multi-junction with MQTT coordination
python adaptive_signal_mqtt.py

# Full system
python app.py
```

### Launch SUMO simulation
```bash
# Windows
run_junction_J1.bat
run_junction_J2.bat
run_junction_J3.bat
```

### View MQTT dashboard
```bash
python mqtt_dashboard.py
```

---

## System Architecture

```
Cloud Backend
├── RL Model Training (train_model.py)
├── TimescaleDB — metrics storage
├── Analytics Dashboard
└── Model Repository (traffic_model.onnx)
        ↕ Daily model updates
Edge Gateway  [<20ms inference · 3.2s swarm consensus]
├── ONNX Runtime — RL policy inference
├── MQTT Broker — inter-junction coordination
├── Sensor Fusion Engine
└── Local Policy Cache → Traffic Signal Actuation
        ↕ Swarm coordination between intersections
Field Sensors
├── Camera detection (emulated)
├── Radar speed sensing (emulated)
├── Loop detector
└── V2X OBU/RSU
```

---

## Research Foundation

- **Swarm Intelligence** — Ant colony optimization, flocking behavior, distributed decision-making
- **Reinforcement Learning** — Q-learning, PPO, multi-agent RL for dynamic signal control
- **Intelligent Transport Systems** — V2X communication, IoT-based sensing, real-time modelling
- **Traffic Engineering** — Queue theory, saturation flow, dynamic phase optimization

---

## Team

| Name | Role |
|------|------|
| Sanjana MS | Team Lead |
| Akshaya Swati R | Team Member |
| Yuvashree M | Team Member |
| Mabel Dannee S | Team Member |
| Varun CM | Mentor · Associate Professor · RMK Engineering College |

---

## Demo

[Watch the live simulation demo](https://drive.google.com/file/d/1gKGzVckTvcz8XcxWt2YfLAiNDfdXXrcP/view?usp=sharing)

The demo shows:
- Side-by-side: Fixed-timer vs Swarm-AI adaptive signal control
- Real-time queue length, throughput, and delay metrics
- Emergency vehicle auto-priority — instant green corridor
- MQTT dashboard showing live junction state coordination

---

## Future Roadmap

- [ ] Small-scale field pilot at 1–2 real intersections
- [ ] AI congestion forecasting — predict jams 10–15 min early
- [ ] Full transit and emergency vehicle priority integration
- [ ] City analytics dashboard with heatmaps and KPIs
- [ ] EV, parking, pedestrian, and cyclist system integration

---

*Built for KPIT Sparkle Techathon · Category: Virtual Engineering and Simulation Driven Systems for Intelligent Vehicle Engineering*
