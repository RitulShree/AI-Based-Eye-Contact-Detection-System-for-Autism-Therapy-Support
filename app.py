from flask import Flask, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/start-test', methods=['GET'])
def start_test():
    try:
        # Run main.py
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True
        )

        output = result.stdout.strip()

        # Return raw output (JSON string from main.py)
        return jsonify({"result": output})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)