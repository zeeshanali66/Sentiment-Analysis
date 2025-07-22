from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
from flask_cors import CORS  # ✅ Import CORS
import os

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # ✅ Allow all origins for now (you can restrict later)

# --- CONFIGURATION ---
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY environment variable not set!")

# Initialize the Hugging Face Inference client
client = InferenceClient(token=HUGGINGFACE_API_KEY)
SENTIMENT_MODEL = "finiteautomata/bertweet-base-sentiment-analysis"

# --- HELPER FUNCTION ---
def analyze_sentiment(user_input):
    """
    Performs sentiment analysis using the Hugging Face Inference API.
    """
    result = client.text_classification(user_input, model=SENTIMENT_MODEL)
    if result and isinstance(result, list) and result[0].get('label'):
        return result[0]['label']
    return "Error: Could not parse model output"

# --- API ROUTES ---
@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "model": SENTIMENT_MODEL,
        "message": "API is ready to receive requests at /detect_emotion"
    })

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_route():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    user_input = data.get('text')

    if not user_input:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    try:
        emotion = analyze_sentiment(user_input)
        return jsonify({"emotion": emotion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
