import matplotlib.pyplot as plt
from data_loader import load_ucihar

X_train, _, _, _, _, _ = load_ucihar()

# Plot les 9 canaux pour le 1er échantillon
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
channels = ["body_acc_x", "body_acc_y", "body_acc_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z"]
for i, ax in enumerate(axes.flat):
    ax.plot(X_train[0, :, i])
    ax.set_title(channels[i])
    ax.grid(True)
plt.tight_layout()
plt.show()