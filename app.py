from flask import Flask, request, render_template
import joblib
import pandas as pd
import re
import os

app = Flask(__name__)

# Load trained model
model = joblib.load(os.path.join('model', 'sentiment_model.pkl'))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Ensure text is string
    return text.lower().strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'csvfile' in request.files and request.files['csvfile'].filename != '':
        file = request.files['csvfile']
        search_term = request.form.get('search_term', '').strip().lower()

        try:
            filename = file.filename.lower()

            if filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                return render_template('index.html', error="Please upload a .csv or .xlsx file.")

            match = None
            for idx, row in df.iterrows():
                row_str = ' '.join(str(value).lower() for value in row.values)
                if search_term in row_str:
                    match = row
                    break

            if match is not None:
                feedback_text = match.get('Feedback', '')
                if pd.isna(feedback_text) or not str(feedback_text).strip():
                    sentiment = 'No feedback provided'
                else:
                    cleaned = preprocess(feedback_text)
                    pred = model.predict([cleaned])[0]
                    sentiment = 'Positive 😀' if pred == 1 else 'Negative 😞'

                return render_template(
                    'index.html',
                    search_result=match.to_dict(),
                    sentiment=sentiment,
                    term=search_term
                )

            # No match found → analyze search term
            cleaned = preprocess(search_term)
            pred = model.predict([cleaned])[0]
            sentiment = 'Positive 😀' if pred == 1 else 'Negative 😞'
            return render_template('index.html', term=search_term, sentiment=sentiment, match_found=False)

        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

    return render_template('index.html', error="Please upload a valid file and search term.")

if __name__ == '__main__':
    app.run(debug=True)
