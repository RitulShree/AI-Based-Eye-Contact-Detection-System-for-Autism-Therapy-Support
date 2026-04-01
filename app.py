from flask import Flask, jsonify, send_from_directory, request
import os
import subprocess
import json
import sys
import csv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "session_log.csv")

app = Flask(__name__)

# -------- Token Counter --------
def get_next_token():
    if not os.path.isfile(LOG_PATH):
        return "TKN-001"
    with open(LOG_PATH, newline='') as f:
        rows = list(csv.DictReader(f))
    return f"TKN-{str(len(rows) + 1).zfill(3)}"

# -------- Serve Kiosk --------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "kiosk.html")

@app.route("/kiosk.css")
def styles():
    return send_from_directory(BASE_DIR, "kiosk.css")

# -------- Serve Reception --------
@app.route("/reception")
def reception():
    return send_from_directory(BASE_DIR, "reception.html")

@app.route("/reception.css")
def reception_css():
    return send_from_directory(BASE_DIR, "reception.css")

# -------- Run Session --------
@app.route('/start-test', methods=['GET'])
def start_test():
    try:
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return jsonify({
                "error": "main.py failed",
                "details": result.stderr.strip() or result.stdout.strip()
            }), 500

        output = result.stdout.strip().splitlines()
        if not output:
            return jsonify({"error": "No output from main.py"}), 500

        report = json.loads(output[-1])
        return jsonify(report)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------- Save Session --------
@app.route('/save-session', methods=['POST'])
def save_session():
    try:
        data = request.get_json()
        token = get_next_token()
        now = datetime.now()
        file_exists = os.path.isfile(LOG_PATH)

        with open(LOG_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "token", "date", "time", "name",
                    "age_group", "prediction", "confidence",
                    "eye_contact", "blink_rate",
                    "fixations", "gaze_variance", "status"
                ])
            writer.writerow([
                token,
                now.strftime("%d-%b-%Y"),
                now.strftime("%H:%M"),
                data.get("name", "Anonymous"),
                data.get("age_group", "-"),
                data.get("prediction"),
                round(data.get("confidence"), 1),
                round(data.get("eye_contact"), 1),
                round(data.get("blink_rate"), 1),
                data.get("fixations"),
                round(data.get("gaze_variance"), 2),
                "Waiting"   # default status
            ])

        return jsonify({"status": "saved", "token": token})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------- Get History (for reception) --------
@app.route('/get-history', methods=['GET'])
def get_history():
    try:
        if not os.path.isfile(LOG_PATH):
            return jsonify([])
        with open(LOG_PATH, newline='') as f:
            rows = list(csv.DictReader(f))
        return jsonify(list(reversed(rows)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------- Mark as Referred --------
@app.route('/refer-patient', methods=['POST'])
def refer_patient():
    try:
        data = request.get_json()
        token = data.get("token")
        doctor = data.get("doctor", "General Physician")

        rows = []
        with open(LOG_PATH, newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row["token"] == token:
                    row["status"] = f"Referred to {doctor}"
                rows.append(row)

        with open(LOG_PATH, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return jsonify({"status": "updated"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
