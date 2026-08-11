from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent

data = np.load(
    PROJECT_DIR / "processed" / "csi_amplitude.npy",
    allow_pickle=True
)

classes = [
    "empty",
    "standing",
    "walking",
    "sitting"
]

for activity in classes:

    records = [
        r for r in data
        if r["activity"] == activity
    ]

    if not records:
        continue

    # Use first recording for initial inspection
    record = records[0]

    amplitude = record["amplitude"]

    # Mean amplitude across subcarriers
    signal = np.mean(
        amplitude,
        axis=1
    )

    plt.figure(figsize=(12, 5))

    plt.plot(signal)

    plt.title(
        f"Wi-Fi CSI Amplitude - {activity.upper()}"
    )

    plt.xlabel("CSI Packet")
    plt.ylabel("Mean CSI Amplitude")

    plt.grid(True)

    plt.tight_layout()

    output = (
        PROJECT_DIR
        / "processed"
        / f"{activity}_csi.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.show()

    print(
        "Saved:",
        output
    )