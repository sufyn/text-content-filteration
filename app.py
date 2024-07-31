import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load the saved model and vectorizer 
loaded_model = joblib.load('logistic_regression_model.pkl')
vect = joblib.load('vectorizer.pkl')  
# vect= TfidfVectorizer(max_features=5000,stop_words='english')

# cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

st.title("Toxicity Prediction App")

user_input = st.text_area("Enter text for toxicity prediction:")

if st.button("Predict"):
    if user_input:
        cleaned_text = [clean_text(text) for text in user_input]

        # Transform the cleaned text
        new_text_vec = vect.transform(cleaned_text)

        predictions = loaded_model.predict_proba(new_text_vec)[:, 1]

        # Print the predicted probabilities as percentages
        for prediction in predictions:
            st.write(f"Abusiveness Percentage: {prediction * 100:.2f}%")
            st.write("Prediction:", prediction)
    else:
        st.warning("Please enter some text.")
