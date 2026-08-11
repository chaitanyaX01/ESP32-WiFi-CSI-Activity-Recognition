import serial
import time
import csv
from pathlib import Path

# ==============================
# SETTINGS
# ==============================

PORT = "/dev/cu.usbmodem1101"
BAUD_RATE = 921600
RECORD_SECONDS = 30

VALID_CLASSES = [
    "empty",
    "standing",
    "walking",
    "sitting"
]

# ==============================
# PROJECT PATH
# ==============================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

# ==============================
# SELECT ACTIVITY
# ==============================

print("\nWi-Fi CSI Dataset Recorder")
print("--------------------------")

for i, name in enumerate(VALID_CLASSES, start=1):
    print(f"{i}. {name}")

choice = input("\nSelect activity number: ").strip()

try:
    choice = int(choice)
    activity = VALID_CLASSES[choice - 1]
except (ValueError, IndexError):
    print("Invalid selection.")
    raise SystemExit

# ==============================
# CREATE OUTPUT FOLDER
# ==============================

activity_dir = DATASET_DIR / activity
activity_dir.mkdir(parents=True, exist_ok=True)

existing = list(activity_dir.glob(f"{activity}_*.csv"))

recording_number = len(existing) + 1

filename = activity_dir / f"{activity}_{recording_number:03d}.csv"

# ==============================
# OPEN SERIAL PORT
# ==============================

print(f"\nOpening {PORT}...")

try:
    ser = serial.Serial(
        PORT,
        BAUD_RATE,
        timeout=1
    )

except serial.SerialException as error:
    print("\nCould not open ESP32 serial port.")
    print(error)
    print("\nCheck the port using:")
    print("ls /dev/cu.*")
    raise SystemExit

time.sleep(2)

ser.reset_input_buffer()

# ==============================
# COUNTDOWN
# ==============================

print(f"\nActivity: {activity}")
print(f"Recording time: {RECORD_SECONDS} seconds")
print(f"Output: {filename}")

print("\nRecording starts in:")

for number in range(3, 0, -1):
    print(number)
    time.sleep(1)

print("\nRECORDING...\n")

# ==============================
# RECORD CSI
# ==============================

start_time = time.time()
packet_count = 0

with open(filename, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "timestamp",
        "elapsed_time",
        "activity",
        "csi_raw"
    ])

    while time.time() - start_time < RECORD_SECONDS:

        line = ser.readline()

        if not line:
            continue

        decoded = line.decode(
            "utf-8",
            errors="ignore"
        ).strip()

        # Save only CSI packets
        if "CSI_DATA" not in decoded:
            continue

        current_time = time.time()
        elapsed = current_time - start_time

        writer.writerow([
            current_time,
            round(elapsed, 4),
            activity,
            decoded
        ])

        packet_count += 1

        print(
            f"\rPackets recorded: {packet_count}",
            end=""
        )

ser.close()

print("\n\nRecording complete!")
print(f"Packets: {packet_count}")
print(f"Saved to:\n{filename}")