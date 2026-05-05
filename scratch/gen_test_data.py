import sys
import os
import numpy as np

# Add src to path to import data_loader
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from data_loader import load_ucihar
except ImportError:
    print("Error: Could not import data_loader")
    sys.exit(1)

def gen_test_header(idx=1):
    _, _, X_test, _, _, _ = load_ucihar()
    sample = X_test[idx]
    print(sample)
    
    header_path = "firmware/test_data.h"
    with open(header_path, "w") as f:
        f.write("#ifndef TEST_DATA_H\n#define TEST_DATA_H\n\n")
        f.write(f"// Sample index {idx} from UCI HAR\n")
        f.write("const float TEST_SAMPLE[128][9] = {\n")
        for row in sample:
            f.write("  {" + ", ".join([f"{v:.6f}f" for v in row]) + "},\n")
        f.write("};\n\n")
        f.write("#endif\n")
    print(f"✅ Generated {header_path}")

if __name__ == "__main__":
    gen_test_header(int(input("index (>1): ")))
