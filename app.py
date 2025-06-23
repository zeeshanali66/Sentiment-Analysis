from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
import os

# Initialize the Flask app
app = Flask(__name__)

# --- CONFIGURATION ---
# Get the Hugging Face API key from Render's environment variables
# This is the secure and correct way to handle secrets.
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Fail fast if the API key is not found during startup.
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY environment variable not set!")

# Initialize the Hugging Face Inference client
client = InferenceClient(token=HUGGINGFACE_API_KEY)

# Define the model you want to use
SENTIMENT_MODEL = "finiteautomata/bertweet-base-sentiment-analysis"

# --- HELPER FUNCTION ---
def analyze_sentiment(user_input):
    """
    Performs sentiment analysis using the Hugging Face Inference API.
    """
    # Use the client to perform text classification
    result = client.text_classification(user_input, model=SENTIMENT_MODEL)
    
    # The result is a list of dictionaries. We want the label from the first one.
    # Example: [{'label': 'POS', 'score': 0.99...}]
    if result and isinstance(result, list) and result[0].get('label'):
        return result[0]['label']
    
    # Return a default or error if the format is unexpected
    return "Error: Could not parse model output"

# --- API ROUTES ---

# <<< CHANGE #1: ADDED THIS ROOT ROUTE >>>
@app.route("/")
def home():
    """
    This is the health check endpoint. It confirms the API is online.
    It solves the '404 Not Found' error when visiting the main URL.
    """
    return jsonify({
        "status": "online",
        "model": SENTIMENT_MODEL,
        "message": "API is ready to receive requests at /detect_emotion"
    })

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_route():
    """
    The main endpoint to handle emotion/sentiment detection.
    """
    # <<< CHANGE #2: ADDED ROBUST ERROR CHECKING >>>
    # Check if the request contains JSON and the 'text' key
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    user_input = data.get('text') # Use .get() to avoid errors if 'text' is missing

    if not user_input:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    try:
        # Perform emotion detection on the user input
        emotion = analyze_sentiment(user_input)

        # Return the detected emotion as JSON
        return jsonify({"emotion": emotion})

    except Exception as e:
        # Handle any other unexpected errors during the API call
        return jsonify({"error": str(e)}), 500

# The if __name__ == '__main__': block is only for local testing.
# Gunicorn (on Render) will not run this block. It imports the 'app' object directly.
if __name__ == '__main__':
    # Use a port from environment or default to 5000 for local dev
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)