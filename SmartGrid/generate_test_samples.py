import os
import sys
import numpy as np

# Add src to path so we can import config and generate_iq
sys.path.append('src')
from config import PROJECT_ROOT  # noqa: E402
from generate_iq import generate_healthy, generate_partial_discharge, generate_arcing, generate_overload  # noqa: E402

# Use a different seed so these are completely new, unseen signals
np.random.seed(888)

output_dir = os.path.join(PROJECT_ROOT, "test_samples")
os.makedirs(output_dir, exist_ok=True)

signals = {
    "Healthy": generate_healthy(),
    "Partial_Discharge": generate_partial_discharge(),
    "Arcing": generate_arcing(),
    "Overload": generate_overload()
}

print("Generating 4 new test samples...")
for cls, signal in signals.items():
    path = os.path.join(output_dir, f"test_{cls}.npy")
    np.save(path, signal)
    print(f"Created: test_samples/test_{cls}.npy")
