"""
ONNX EXPORTER - Convert PyTorch Model to Edge-Ready Format

Converts trained PyTorch model to ONNX format for:
- Hardware-independent deployment
- Edge device compatibility (Jetson, Raspberry Pi, IPCs)
- Optimized inference performance
"""

import torch
import torch.nn as nn
import pickle
import onnx
import onnxruntime as ort
import numpy as np

# ============================================================================
# MODEL ARCHITECTURE (must match training)
# ============================================================================
class TrafficSignalNN(nn.Module):
    """Same architecture as training script"""
    
    def __init__(self, input_size=9, hidden1=64, hidden2=32, num_phases=4):
        super(TrafficSignalNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, num_phases)
        )
        
    def forward(self, x):
        return self.network(x)

# ============================================================================
# ONNX EXPORT FUNCTION
# ============================================================================
def export_to_onnx():
    """Export PyTorch model to ONNX format"""
    
    print("=" * 80)
    print("ONNX EXPORT - Converting PyTorch to Edge-Ready Format")
    print("=" * 80)
    
    # Load model configuration
    print("\n[1/5] Loading model configuration...")
    try:
        with open('model_config.pkl', 'rb') as f:
            config = pickle.load(f)
        print(f"✓ Config loaded:")
        print(f"    Input size: {config['input_size']}")
        print(f"    Hidden layers: {config['hidden1']}, {config['hidden2']}")
        print(f"    Output phases: {config['num_phases']}")
    except FileNotFoundError:
        print("✗ model_config.pkl not found")
        print("  Run train_model.py first")
        return
    
    # Recreate model architecture
    print("\n[2/5] Recreating model architecture...")
    model = TrafficSignalNN(
        input_size=config['input_size'],
        hidden1=config['hidden1'],
        hidden2=config['hidden2'],
        num_phases=config['num_phases']
    )
    print("✓ Model architecture created")
    
    # Load trained weights
    print("\n[3/5] Loading trained weights...")
    try:
        model.load_state_dict(torch.load('traffic_model.pth'))
        model.eval()
        print("✓ Weights loaded successfully")
    except FileNotFoundError:
        print("✗ traffic_model.pth not found")
        print("  Run train_model.py first")
        return
    
    # Export to ONNX
    print("\n[4/5] Exporting to ONNX format...")
    
    dummy_input = torch.randn(1, config['input_size'])
    onnx_file = "traffic_model.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Model exported to: {onnx_file}")
    
    # Validate ONNX model
    print("\n[5/5] Validating ONNX model...")
    
    try:
        onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed")
    except Exception as e:
        print(f"✗ ONNX validation failed: {e}")
        return
    
    # Test inference with ONNX Runtime
    print("\n[BONUS] Testing ONNX Runtime inference...")
    
    try:
        ort_session = ort.InferenceSession(onnx_file)
        
        test_input = np.random.randn(1, config['input_size']).astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        predicted_phase = np.argmax(ort_outputs[0], axis=1)[0]
        
        print(f"✓ ONNX Runtime test successful")
        print(f"    Test input shape: {test_input.shape}")
        print(f"    Predicted phase: {predicted_phase}")
        
    except Exception as e:
        print(f"⚠ ONNX Runtime test failed: {e}")
    
    # Print summary
    print()
    print("=" * 80)
    print("EXPORT SUMMARY")
    print("=" * 80)
    print(f"✓ PyTorch model: traffic_model.pth")
    print(f"✓ ONNX model: {onnx_file}")
    print(f"✓ Model size: {get_file_size(onnx_file)}")
    print()
    print("DEPLOYMENT TARGETS:")
    print("  • Edge devices (Jetson, Raspberry Pi)")
    print("  • Industrial PCs")
    print("  • Microcontrollers (with ONNX Runtime support)")
    print("  • Cloud inference servers")
    print()
    print("NEXT STEPS:")
    print("  1. Copy traffic_model.onnx to your edge device")
    print("  2. Run adaptive_signal.py with ONNX inference")
    print("  3. Monitor performance and collect feedback")
    print("=" * 80)
    print()

def get_file_size(filepath):
    """Get human-readable file size"""
    import os
    size_bytes = os.path.getsize(filepath)
    
    for unit in ['B', 'KB', 'MB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} GB"

# ============================================================================
# VERIFICATION SCRIPT
# ============================================================================
def compare_pytorch_onnx():
    """Compare PyTorch and ONNX outputs to ensure consistency"""
    
    print("\n" + "=" * 80)
    print("VERIFICATION - Comparing PyTorch vs ONNX Outputs")
    print("=" * 80)
    
    with open('model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    pytorch_model = TrafficSignalNN(
        input_size=config['input_size'],
        hidden1=config['hidden1'],
        hidden2=config['hidden2'],
        num_phases=config['num_phases']
    )
    pytorch_model.load_state_dict(torch.load('traffic_model.pth'))
    pytorch_model.eval()
    
    ort_session = ort.InferenceSession('traffic_model.onnx')
    
    num_tests = 5
    print(f"\nRunning {num_tests} comparison tests...\n")
    
    for i in range(num_tests):
        test_input_np = np.random.randn(1, config['input_size']).astype(np.float32)
        test_input_torch = torch.tensor(test_input_np)
        
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input_torch).numpy()
        pytorch_phase = np.argmax(pytorch_output)
        
        ort_inputs = {ort_session.get_inputs()[0].name: test_input_np}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        onnx_phase = np.argmax(onnx_output)
        
        diff = np.abs(pytorch_output - onnx_output).max()
        match = "✓" if pytorch_phase == onnx_phase else "✗"
        
        print(f"Test {i+1}:")
        print(f"  PyTorch phase: {pytorch_phase} | ONNX phase: {onnx_phase} {match}")
        print(f"  Max difference: {diff:.6f}")
    
    print()
    print("✓ Verification complete")
    print("=" * 80)

if __name__ == "__main__":
    export_to_onnx()
    
    print("\nRun verification? (y/n): ", end="")
    try:
        response = input().strip().lower()
        if response == 'y':
            compare_pytorch_onnx()
    except:
        pass
