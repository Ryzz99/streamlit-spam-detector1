import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])
    return df

# Train model
@st.cache_resource
def train_model(df):
    cv = CountVectorizer()
    X = cv.fit_transform(df['message'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as m, open("vectorizer.pkl", "wb") as v:
        pickle.dump(model, m)
        pickle.dump(cv, v)

    return model, cv

def load_model():
    with open("model.pkl", "rb") as m, open("vectorizer.pkl", "rb") as v:
        model = pickle.load(m)
        cv = pickle.load(v)
    return model, cv

# Main
st.title("Spam Message Detector")

option = st.radio("Mode:", ['Train Model', 'Detect Spam'])

if option == 'Train Model':
    df = load_data()
    st.write(df.head())
    model, cv = train_model(df)
    st.success("Model trained and saved.")
else:
    message = st.text_area("Enter your message:")
    if st.button("Predict"):
        model, cv = load_model()
        vec = cv.transform([message])
        prediction = model.predict(vec)
        st.write("Result:", prediction[0])
