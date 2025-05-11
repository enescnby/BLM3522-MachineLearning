from flask import Flask, request, jsonify
import joblib
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))

model = joblib.load('NB_topic_classification_model.pkl')
vectorizer = joblib.load('tf_idf_vectorizer.pkl')

def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    words = text.split()
    words = [w for w in words if not w in stop_words]
    return ' '.join(words)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    raw_text = data.get("text", "")
    cleaned = preprocess_text(raw_text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)
    return jsonify({'prediction': pred[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
