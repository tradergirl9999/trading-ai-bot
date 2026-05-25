from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = """
You are Girltrader's AI trading assistant.

Analyze market structure using:
- MSNR
- liquidity sweeps
- HTF bias
- displacement
- rejection wicks
- tight SL logic

If setup is weak, say NO TRADE.

Return:
Bias
Entry
SL
TP1
TP2
Confidence
Reason
"""

@app.route("/")
def home():
    return "Trading AI Bot Running"

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json

    prompt = f"""
    TradingView Alert Data:
    {data}

    Analyze and return trading signal.
    """

    response = client.responses.create(
        model="gpt-5.5",
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}
        ]
    )

    signal = response.output_text

    print(signal)

    return jsonify({
        "signal": signal
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
