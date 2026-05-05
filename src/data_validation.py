import os
import numpy as np

BASE = "data/UCI_HAR_dataset"

def check_file(path, expected_shape_2d=None):
    if not os.path.exists(path):
        print(f"❌ Manquant : {path}")
        return False
    data = np.loadtxt(path)
    print(f"✅ {os.path.basename(path):30s} → shape {data.shape}")
    if expected_shape_2d and data.shape[1] != expected_shape_2d:
        print(f"   ⚠️  Attendu 2ème dim = {expected_shape_2d}, obtenu {data.shape[1]}")
        return False
    return True

print("🔍 Vérification des signaux bruts (Inertial Signals)...")
signals = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z"
]
all_ok = True
for split in ["train", "test"]:
    print(f"\n[{split.upper()}]")
    for sig in signals:
        path = os.path.join(BASE, split, "Inertial Signals", f"{sig}_{split}.txt")
        ok = check_file(path, expected_shape_2d=128)
        all_ok = all_ok and ok
    
    y_path = os.path.join(BASE, split, f"y_{split}.txt")
    ok = check_file(y_path)
    all_ok = all_ok and ok

if all_ok:
    print("\n🎉 Tout est bon ! Tu peux utiliser le loader tel quel.")
else:
    print("\n⚠️  Il y a des fichiers manquants ou des shapes inattendues.")