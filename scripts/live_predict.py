import serial
import time
import re
import json
import joblib
import numpy as np
from pathlib import Path


# ==========================================
# SETTINGS
# ==========================================

PORT = "/dev/cu.usbmodem1101"
BAUD_RATE = 921600

WINDOW_SECONDS = 2.0

CLASS_NAMES = [
    "EMPTY",
    "STANDING",
    "WALKING",
    "SITTING"
]


# ==========================================
# PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "random_forest_csi.joblib"
)


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully")


# Number of features model expects
number_features = model.n_features_in_

# We used 5 feature groups:
# mean, std, median, range, rms
EXPECTED_SUBCARRIERS = number_features // 5

print(
    "Expected subcarriers:",
    EXPECTED_SUBCARRIERS
)


# ==========================================
# EXTRACT CSI
# ==========================================

def extract_csi(line):

    if "CSI_DATA" not in line:
        return None

    match = re.search(
        r'(\[[^\]]+\])',
        line
    )

    if not match:
        return None

    try:

        values = json.loads(
            match.group(1)
        )

    except Exception:
        return None

    values = np.array(
        values,
        dtype=float
    )

    if len(values) < 2:
        return None

    if len(values) % 2 != 0:
        values = values[:-1]

    return values


# ==========================================
# CSI AMPLITUDE
# ==========================================

def calculate_amplitude(values):

    first = values[0::2]
    second = values[1::2]

    amplitude = np.sqrt(
        first ** 2
        +
        second ** 2
    )

    return amplitude


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(window):

    mean = np.mean(
        window,
        axis=0
    )

    std = np.std(
        window,
        axis=0
    )

    median = np.median(
        window,
        axis=0
    )

    minimum = np.min(
        window,
        axis=0
    )

    maximum = np.max(
        window,
        axis=0
    )

    signal_range = (
        maximum - minimum
    )

    rms = np.sqrt(
        np.mean(
            window ** 2,
            axis=0
        )
    )

    features = np.concatenate([
        mean,
        std,
        median,
        signal_range,
        rms
    ])

    return features


# ==========================================
# SERIAL CONNECTION
# ==========================================

print("\nOpening ESP32 serial port...")

try:

    ser = serial.Serial(
        PORT,
        BAUD_RATE,
        timeout=1
    )

except Exception as error:

    print("\nCould not open ESP32.")
    print(error)

    print("\nCheck port with:")
    print("ls /dev/cu.*")

    raise SystemExit


time.sleep(2)

ser.reset_input_buffer()

print("\nESP32 connected!")
print("=" * 55)

print(
    "\nREAL-TIME Wi-Fi CSI ACTIVITY DETECTION"
)

print("=" * 55)


# ==========================================
# LIVE PREDICTION
# ==========================================

try:

    while True:

        amplitudes = []

        start_time = time.time()

        # ----------------------------------
        # COLLECT 2 SECONDS CSI
        # ----------------------------------

        while (
            time.time() - start_time
            <
            WINDOW_SECONDS
        ):

            raw = ser.readline()

            if not raw:
                continue

            line = raw.decode(
                "utf-8",
                errors="ignore"
            ).strip()

            values = extract_csi(line)

            if values is None:
                continue

            amplitude = calculate_amplitude(
                values
            )

            # Ensure same subcarrier count
            # as training dataset
            if len(amplitude) < EXPECTED_SUBCARRIERS:
                continue

            amplitude = amplitude[
                :EXPECTED_SUBCARRIERS
            ]

            amplitudes.append(
                amplitude
            )


        # ----------------------------------
        # CHECK WINDOW
        # ----------------------------------

        if len(amplitudes) < 20:

            print(
                "Not enough CSI packets:",
                len(amplitudes)
            )

            continue


        window = np.array(
            amplitudes,
            dtype=np.float32
        )


        # ----------------------------------
        # FEATURES
        # ----------------------------------

        features = extract_features(
            window
        )

        features = features.reshape(
            1,
            -1
        )


        # ----------------------------------
        # PREDICT
        # ----------------------------------

        prediction = model.predict(
            features
        )[0]

        probabilities = model.predict_proba(
            features
        )[0]

        confidence = (
            probabilities[prediction]
            * 100
        )


        # ----------------------------------
        # OUTPUT
        # ----------------------------------

        print("\n" + "-" * 55)

        print(
            "Activity :",
            CLASS_NAMES[prediction]
        )

        print(
            f"Confidence: "
            f"{confidence:.2f}%"
        )

        print(
            "CSI packets:",
            len(amplitudes)
        )

        print("-" * 55)


except KeyboardInterrupt:

    print(
        "\n\nStopping real-time detection..."
    )


finally:

    ser.close()

    print(
        "Serial connection closed."
    )