from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/search_resumes', methods=['GET'])
def search_resumes():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    try:
        # Call the search_resumes.py script with the --query argument
        result = subprocess.run(
            ['python3', 'search_resumes.py', '--query', query],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return jsonify({"error": "Error executing search_resumes.py", "details": result.stderr}), 500

        return jsonify({"result": result.stdout.strip()})
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)