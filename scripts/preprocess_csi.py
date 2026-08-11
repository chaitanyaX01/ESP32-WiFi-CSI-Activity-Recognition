from pathlib import Path
import pandas as pd
import numpy as np
import json
import re

# ==========================================
# PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
PROCESSED_DIR = PROJECT_DIR / "processed"

PROCESSED_DIR.mkdir(exist_ok=True)

CLASSES = [
    "empty",
    "standing",
    "walking",
    "sitting"
]


# ==========================================
# EXTRACT CSI ARRAY
# ==========================================

def extract_csi(raw_line):

    if not isinstance(raw_line, str):
        return None

    # Get the final [...] section from CSI_DATA line
    match = re.search(r'(\[[^\]]+\])', raw_line)

    if not match:
        return None

    try:
        values = json.loads(match.group(1))
    except:
        return None

    values = np.array(values, dtype=float)

    # CSI data must contain I/Q pairs
    if len(values) < 2:
        return None

    # Remove last value if odd length
    if len(values) % 2 != 0:
        values = values[:-1]

    return values


# ==========================================
# I/Q -> AMPLITUDE
# ==========================================

def calculate_amplitude(values):

    first = values[0::2]
    second = values[1::2]

    amplitude = np.sqrt(
        first ** 2 + second ** 2
    )

    return amplitude


# ==========================================
# PROCESS DATASET
# ==========================================

all_records = []

print("\nCSI PREPROCESSING")
print("=" * 60)

for activity in CLASSES:

    folder = DATASET_DIR / activity

    files = sorted(folder.glob("*.csv"))

    print(f"\nProcessing: {activity}")
    print("-" * 60)

    for file in files:

        try:

            df = pd.read_csv(file)

            amplitudes = []
            timestamps = []

            for _, row in df.iterrows():

                raw = row.get("csi_raw")

                values = extract_csi(raw)

                if values is None:
                    continue

                amplitude = calculate_amplitude(values)

                amplitudes.append(amplitude)

                timestamps.append(
                    row.get("elapsed_time", 0)
                )

            if len(amplitudes) == 0:

                print(
                    f"{file.name:<25} NO VALID CSI"
                )

                continue

            # Make all packets same number of subcarriers
            minimum_length = min(
                len(x) for x in amplitudes
            )

            amplitudes = np.array(
                [
                    x[:minimum_length]
                    for x in amplitudes
                ]
            )

            print(
                f"{file.name:<25}"
                f"{len(amplitudes):>6} packets   "
                f"{minimum_length} subcarriers"
            )

            all_records.append({
                "activity": activity,
                "filename": file.name,
                "amplitude": amplitudes,
                "timestamps": np.array(timestamps)
            })

        except Exception as error:

            print(
                f"{file.name:<25} ERROR: {error}"
            )


# ==========================================
# SAVE
# ==========================================

output = PROCESSED_DIR / "csi_amplitude.npy"

np.save(
    output,
    np.array(all_records, dtype=object),
    allow_pickle=True
)

print("\n" + "=" * 60)
print("PROCESSING COMPLETE")

print(
    "Valid recordings:",
    len(all_records)
)

print(
    "Saved:",
    output
)