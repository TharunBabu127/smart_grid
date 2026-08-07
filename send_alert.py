import serial
import time
import subprocess
import os
import sys
import threading
import itertools

# 1. SETUP THE CONNECTION TO ARDUINO
arduino_port = 'COM5' 

print(f"Connecting to Arduino on {arduino_port}...")
try:
    uno = serial.Serial(arduino_port, 115200, timeout=1)
    time.sleep(2)
    print("Successfully connected to Arduino!\n")
except Exception as e:
    print(f"ERROR: Could not connect. Is the Arduino plugged in and Serial Monitor closed? \nDetails: {e}")
    exit()

# 2. GET THE IQ SAMPLE LOCATION
iq_path = input("Paste the location of the IQ sample: ")
print(f"\nAnalyzing {iq_path}...\n")

# --- Animation Logic ---
done = False
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\r[ ' + c + ' ] Running CNN Model & Extracting Features...')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r[ * ] CNN Model Execution Complete!                   \n')

t = threading.Thread(target=animate)
t.start()
# -----------------------

# 3. RUN YOUR CNN MODEL
try:
    process = subprocess.Popen(
        ['python', 'src/realtime.py'],
        cwd='SmartGrid',
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate(input=iq_path + '\n')
    
    # Stop animation
    done = True
    t.join()

    # Parse output
    fault_detected = "Unknown"
    confidence = "Unknown"
    
    for line in stdout.split('\n'):
        if "Predicted Fault Type :" in line:
            fault_detected = line.split(":")[-1].strip()
        elif "Confidence           :" in line:
            confidence = line.split(":")[-1].strip()
            
    if fault_detected == "Unknown":
        print("Error: Could not parse prediction from realtime.py output.")
        print("Output was:\n", stdout)
            
except Exception as e:
    done = True
    t.join()
    print(f"Error running model: {e}")
    fault_detected = "Error"
    confidence = "N/A"

print(f"\n--- PREDICTION RESULTS ---")
print(f"Model detected : {fault_detected}")
print(f"Confidence     : {confidence}")
print(f"--------------------------\n")


# 4. SEND THE RESULT TO ARDUINO
if fault_detected == "Healthy":
    print("✅ System is Healthy. No SMS alert needed.")
elif fault_detected in ["Unknown", "Error"]:
    print("⚠️ Error in detection. No SMS alert sent.")
else:
    print(f"🚨 Fault detected! Sending SMS alert to engineer...")
    location = "Sector 7"
    antenna = "RF-42"
    
    # Example format: Overload, Location: Sector 7, Antenna: RF-42
    message_to_send = f"{fault_detected}, Location: {location}, Antenna: {antenna}\n"
    uno.write(message_to_send.encode())
    
    # 5. WAIT FOR ARDUINO TO PROCESS
    print("Waiting for Arduino and GSM900A to process the SMS. Reading Arduino output...")
    timeout = time.time() + 7 # wait up to 7 seconds
    while time.time() < timeout:
        if uno.in_waiting > 0:
            arduino_response = uno.readline().decode('utf-8', errors='ignore').strip()
            if arduino_response:
                print(f"[ARDUINO SAYS]: {arduino_response}")
        time.sleep(0.1)
    
    print("✅ Alert sequence finished!")

uno.close()
print("\nConnection closed.")
