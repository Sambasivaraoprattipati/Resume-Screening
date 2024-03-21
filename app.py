from flask import Flask, request, render_template, redirect, url_for
import os
import pickle
from PyPDF2 import PdfReader
import docx
import textract
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords data
nltk.download('stopwords')

app = Flask(__name__)


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into a string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

UPLOAD_FOLDER = 'Uploaded_Resumes'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Function to extract text from different file types
def extract_text(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension.lower() == '.docx':
        return extract_text_from_docx(file_path)
    else:
        return extract_text_from_other(file_path)

def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as f:
        reader = PdfReader(f)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text

def extract_text_from_other(file_path):
    return textract.process(file_path).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    # Handle the about page logic here
    return render_template('about.html', result="Positive")

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        feedback = request.form['feedback']
        return 'Thank you for your feedback!'
    else:
        return render_template('feedback.html', result="Positive")
category_mapping = {
    17: 'Data Science',
    11: 'HR',
    15: 'Advocate',
    3: 'Arts',
    4: 'Web Designing',
    5: 'Mechanical Engineer',
    6: 'Sales',
    7: 'Health and fitness',
    8: 'Civil Engineer',
    9: 'Java Developer',
    10: 'Business Analyst',
    1: 'SAP Developer',
    12: 'Automation Testing',
    13: 'Electrical Engineering',
    14: 'Operations Manager',
    2: 'Python Developer',
    16: 'DevOps Engineer',
    0: 'Network Security Engineer',
    18: 'PMO',
    19: 'Database',
    20: 'Hadoop',
    21: 'ETL Developer',
    22: 'DotNet Developer',
    23: 'Blockchain',
    24: 'Testing'
}
@app.route('/upload', methods=['POST'])
def upload_resume():
    if request.method == 'POST':
        file = request.files['resume']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            # Extract text from the uploaded file
            try:
                resume_text = extract_text(filename)
            except Exception as e:
                # Handle extraction error
                return f"Error: {e}"
            
            # Preprocess the text
            cleaned_resume_text = preprocess_text(resume_text)
            
            # Fit and transform the text data using TF-IDF vectorizer
            X = tfidf_vectorizer.fit_transform([cleaned_resume_text])
            
            # Ensure the dimensionality of X matches the model's expectations
            X = np.pad(X.toarray(), ((0, 0), (0, 7351 - X.shape[1])), mode='constant')
            
            # Make predictions
            predicted_category = model.predict(X)
            predicted_category_name = category_mapping.get(predicted_category[0], "Unknown")
            
            # Redirect to the result route with category name
            return redirect(url_for('result', category=predicted_category_name))
    return render_template('index.html')




@app.route('/result/<category>')
def result(category):
    return render_template('result.html', category=category)

if __name__ == '__main__':
    app.run(debug=True)
