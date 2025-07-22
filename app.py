from flask import Flask, render_template, request
import numpy as np
import pickle
import re
import nltk
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Constants
MAX_LEN = 100  # Must match training
app = Flask(__name__)

# Load model, tokenizer, label encoder
model = load_model("gru_complaint_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Add variation function
def add_variation(text):
    text = text.lower()
    variations = [
        text,
        text.replace("deducted", "charged"),
        text.replace("failed", "not successful"),
        text.replace("refund", "money return"),
        text + " please look into this",
        "issue: " + text,
        "complaint: " + text
    ]
    return random.choice(variations)

# Web route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        user_input = request.form["complaint"]

        # Step 1: Add variation
        varied_text = add_variation(user_input)

        # Step 2: Clean
        cleaned = clean_text(varied_text)

        # Step 3: Tokenize & pad
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        # Step 4: Predict
        pred = model.predict(padded)
        class_index = np.argmax(pred, axis=1)[0]
        prediction = label_encoder.inverse_transform([class_index])[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
