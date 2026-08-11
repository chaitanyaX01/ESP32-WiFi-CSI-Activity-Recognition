from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATASET = PROJECT / "dataset"

classes = ["empty", "standing", "walking", "sitting"]

print("\nWi-Fi CSI Dataset Check")
print("=" * 50)

total_files = 0
total_packets = 0
bad_files = []

for activity in classes:

    folder = DATASET / activity
    files = sorted(folder.glob("*.csv"))

    print(f"\n{activity.upper()}")
    print("-" * 50)

    class_packets = 0

    for file in files:

        try:
            df = pd.read_csv(file)

            packets = len(df)

            class_packets += packets
            total_packets += packets
            total_files += 1

            status = "OK"

            if packets == 0:
                status = "EMPTY"
                bad_files.append(file)

            print(
                f"{file.name:<25}"
                f"{packets:>6} packets   {status}"
            )

        except Exception as e:

            print(f"{file.name:<25} ERROR")

            bad_files.append(file)

    print(f"\nRecordings : {len(files)}")
    print(f"Packets    : {class_packets}")

print("\n" + "=" * 50)

print("Total recordings:", total_files)
print("Total CSI packets:", total_packets)

if bad_files:

    print("\nProblem files:")

    for file in bad_files:
        print(file)

else:
    print("\nDataset looks good!")