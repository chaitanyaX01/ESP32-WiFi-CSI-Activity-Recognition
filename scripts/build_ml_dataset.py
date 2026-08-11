from pathlib import Path
import numpy as np
import pandas as pd

# =====================================
# PATHS
# =====================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "processed"

DATA_FILE = PROCESSED_DIR / "csi_amplitude.npy"

# =====================================
# SETTINGS
# =====================================

WINDOW_SECONDS = 2.0
STEP_SECONDS = 1.0

MIN_PACKETS = 20

CLASSES = [
    "empty",
    "standing",
    "walking",
    "sitting"
]

LABELS = {
    "empty": 0,
    "standing": 1,
    "walking": 2,
    "sitting": 3
}

RANDOM_SEED = 42

# =====================================
# LOAD DATA
# =====================================

data = np.load(
    DATA_FILE,
    allow_pickle=True
)

rng = np.random.default_rng(RANDOM_SEED)

# =====================================
# FEATURE EXTRACTION
# =====================================

def extract_features(window):

    # window shape:
    # packets × subcarriers

    mean = np.mean(window, axis=0)

    std = np.std(window, axis=0)

    median = np.median(window, axis=0)

    minimum = np.min(window, axis=0)

    maximum = np.max(window, axis=0)

    signal_range = maximum - minimum

    rms = np.sqrt(
        np.mean(window ** 2, axis=0)
    )

    # Combine all feature groups
    features = np.concatenate([
        mean,
        std,
        median,
        signal_range,
        rms
    ])

    return features


# =====================================
# CREATE WINDOWS
# =====================================

def create_windows(record):

    amplitude = record["amplitude"]

    timestamps = np.array(
        record["timestamps"],
        dtype=float
    )

    if len(timestamps) < MIN_PACKETS:
        return []

    # Start timestamps from zero
    timestamps = timestamps - timestamps[0]

    windows = []

    start_time = 0

    end_time = timestamps[-1]

    while start_time + WINDOW_SECONDS <= end_time:

        mask = (
            (timestamps >= start_time)
            &
            (timestamps < start_time + WINDOW_SECONDS)
        )

        window = amplitude[mask]

        if len(window) >= MIN_PACKETS:

            features = extract_features(window)

            windows.append(features)

        start_time += STEP_SECONDS

    return windows


# =====================================
# STORAGE
# =====================================

splits = {
    "train": {
        "X": [],
        "y": [],
        "meta": []
    },
    "val": {
        "X": [],
        "y": [],
        "meta": []
    },
    "test": {
        "X": [],
        "y": [],
        "meta": []
    }
}


# =====================================
# PROCESS EACH CLASS
# =====================================

print("\nBUILDING ML DATASET")
print("=" * 60)

for activity in CLASSES:

    records = [
        r for r in data
        if r["activity"] == activity
    ]

    print(
        f"\n{activity.upper()} recordings:",
        len(records)
    )

    if len(records) < 20:

        print(
            "WARNING: expected 20 recordings"
        )

    # Randomize recording order
    indices = np.arange(len(records))

    rng.shuffle(indices)

    shuffled = [
        records[i]
        for i in indices
    ]

    # ---------------------------------
    # Recording-level split
    # ---------------------------------

    train_records = shuffled[:14]

    val_records = shuffled[14:17]

    test_records = shuffled[17:20]

    split_groups = {
        "train": train_records,
        "val": val_records,
        "test": test_records
    }

    for split_name, group in split_groups.items():

        for record in group:

            windows = create_windows(record)

            label = LABELS[activity]

            for window_number, features in enumerate(windows):

                splits[split_name]["X"].append(
                    features
                )

                splits[split_name]["y"].append(
                    label
                )

                splits[split_name]["meta"].append({
                    "activity": activity,
                    "filename": record["filename"],
                    "window": window_number
                })

            print(
                f"{split_name:<6}"
                f" {record['filename']:<25}"
                f"{len(windows):>4} windows"
            )


# =====================================
# SAVE FILES
# =====================================

for split_name in [
    "train",
    "val",
    "test"
]:

    X = np.array(
        splits[split_name]["X"],
        dtype=np.float32
    )

    y = np.array(
        splits[split_name]["y"],
        dtype=np.int32
    )

    np.save(
        PROCESSED_DIR / f"X_{split_name}.npy",
        X
    )

    np.save(
        PROCESSED_DIR / f"y_{split_name}.npy",
        y
    )

    metadata = pd.DataFrame(
        splits[split_name]["meta"]
    )

    metadata.to_csv(
        PROCESSED_DIR / f"{split_name}_metadata.csv",
        index=False
    )

    print("\n" + "-" * 60)

    print(split_name.upper())

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )


print("\n" + "=" * 60)

print("ML DATASET CREATED")

print("\nSaved in:")

print(PROCESSED_DIR)
