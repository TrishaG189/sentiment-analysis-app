import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    This app predicts whether a movie review is
    Positive or Negative using:

    • TF-IDF Vectorization
    • Logistic Regression
    • Scikit-Learn
    """
)

# Main page
st.title("🎬 Movie Review Sentiment Analyzer")
st.write(
    "Analyze whether a movie review is positive or negative using Machine Learning."
)

review = st.text_area(
    "Enter your review:",
    height=150,
    placeholder="Example: This movie was absolutely amazing!"
)

if st.button("Analyze Sentiment"):

    if review.strip():

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)[0]

        probability = model.predict_proba(review_vector)

        confidence = max(probability[0])

        if prediction == "positive":
            st.success("😊 Positive Review")
            st.balloons()
        else:
            st.error("😞 Negative Review")

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2%}"
        )

    else:
        st.warning("Please enter a review.")