"""
Flask API wrapper for the RAG Chat App (Retail).

Exposes:
  GET  /health            – liveness / readiness probe
  POST /api/chat          – chat with products (RAG)
  GET  /                  – basic info page

Request body for /api/chat:
{
  "messages": [{"role": "user", "content": "I need a tent for 4 people"}],
  "context": {}           // optional overrides
}
"""

import os
import json
from flask import Flask, request, jsonify
from chat_with_products import chat_with_products

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "app": "rag-chatapp-retail",
        "description": "RAG Chat App for retail product recommendations",
        "endpoints": {
            "POST /api/chat": "Send chat messages to get product recommendations",
            "GET /health": "Health check",
        },
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    messages = body.get("messages", [])
    context = body.get("context", {})

    if not messages:
        return jsonify({"error": "messages array is required"}), 400

    try:
        result = chat_with_products(messages=messages, context=context)
        # result["message"] is an Azure SDK ChatResponseMessage; serialise it
        msg = result["message"]
        return jsonify({
            "message": {
                "role": msg.role if hasattr(msg, "role") else "assistant",
                "content": msg.content if hasattr(msg, "content") else str(msg),
            },
            "context": result.get("context", {}),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
