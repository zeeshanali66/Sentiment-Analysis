from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
import os

# Initialize the Flask app
app = Flask(__name__)

# Use the Hugging Face API key directly in your code
client = InferenceClient(
    provider="hf-inference",
    api_key="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",  # Replace with your actual API key
)

# Function to perform sentiment analysis (emotion detection)
def analyze_sentiment(user_input):
    result = client.text_classification(
        user_input,  # Example input text (dynamic)
        model="finiteautomata/bertweet-base-sentiment-analysis",  # Model for sentiment analysis
    )
    
    # Extract only the emotion label (e.g., POSITIVE, NEGATIVE) from the result
    emotion = result[0]['label']
    return emotion

# API endpoint to handle emotion detection
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        # Get the text from the request body
        data = request.get_json()
        user_input = data['text']  # Extract the text

        # Perform emotion detection on the user input
        emotion = analyze_sentiment(user_input)

        # Return the detected emotion as JSON
        return jsonify({"emotion": emotion})

    except Exception as e:
        # Handle any errors and return an error message
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
