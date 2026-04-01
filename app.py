from flask import Flask, jsonify, request, send_from_directory
import csv
from datetime import datetime
import os
import subprocess
import json
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/styles.css")
def styles():
    return send_from_directory(BASE_DIR, "styles.css")

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
            return jsonify({"error": "No output received from main.py"}), 500

        report_line = output[-1]
        report = json.loads(report_line)

        return jsonify(report)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/save-session', methods=['POST'])
def save_session():
    try:
        data = request.get_json()
        log_path = os.path.join(BASE_DIR, "session_log.csv")
        file_exists = os.path.isfile(log_path)

        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "date", "time", "name",
                    "age_group", "prediction",
                    "confidence", "eye_contact",
                    "blink_rate", "fixations", "gaze_variance"
                ])
            now = datetime.now()
            writer.writerow([
                now.strftime("%d-%b-%Y"),
                now.strftime("%H:%M"),
                data.get("name", "Anonymous"),
                data.get("age_group", "-"),
                data.get("prediction"),
                round(data.get("confidence"), 1),
                round(data.get("eye_contact"), 1),
                round(data.get("blink_rate"), 1),
                data.get("fixations"),
                round(data.get("gaze_variance"), 2)
            ])
        return jsonify({"status": "saved"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-history', methods=['GET'])
def get_history():
    try:
        log_path = os.path.join(BASE_DIR, "session_log.csv")
        if not os.path.isfile(log_path):
            return jsonify([])

        sessions = []
        with open(log_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sessions.append(row)

        return jsonify(list(reversed(sessions)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
