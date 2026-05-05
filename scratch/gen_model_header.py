import os

def tflite_to_header(model_path, header_path, var_name="model_tflite"):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        return
    with open(model_path, "rb") as f:
        data = f.read()
    
    with open(header_path, "w") as f:
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        f.write("// Model quantized with INT8\n")
        f.write(f"const unsigned char {var_name}[] = {{\n  ")
        for i, b in enumerate(data):
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n  ")
        f.write("\n};\n")
        f.write(f"const unsigned int {var_name}_len = {len(data)};\n\n")
        f.write("#endif\n")

if __name__ == "__main__":
    tflite_to_header("models/har_int8_3_target_sit.tflite", "firmware/model_data.h")
    print("✅ firmware/model_data.h generated")
