import serial

PORT = "/dev/cu.usbmodem1101"

BAUD = 115200

ser = serial.Serial(PORT, BAUD)

while True:
    line = ser.readline().decode(errors="ignore").strip()

    print(line)