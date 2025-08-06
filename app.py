import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter

# Download stopwords only once
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load ML model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text_vectorized = vectorizer.transform([text])
    sentiment = model.predict(text_vectorized)
    return "Negative" if sentiment == 0 else "Positive"

# Initialize Nitter for scraping tweets
@st.cache_resource
def initialize_scraper():
    return Nitter(log_level=1)

# Generate a colored card
def create_card(tweet_text, sentiment):
    color = "#28a745" if sentiment == "Positive" else "#dc3545"
    return f"""
        <div style="background-color: {color}; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <strong style="color: white;">{sentiment} Sentiment</strong>
            <p style="color: white;">{tweet_text}</p>
        </div>
    """

# Streamlit main app
def main():
    st.title("💬 Twitter Sentiment Analysis")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()

    option = st.radio("Choose an option", ["Input text", "Get tweets from user"])

    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment:")
        if st.button("Analyze"):
            if text_input.strip() != "":
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                st.success(f"Sentiment: {sentiment}")
            else:
                st.warning("Please enter some text.")

    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter username (without @):")
        if st.button("Fetch Tweets"):
            if username.strip() != "":
                tweets_data = scraper.get_tweets(username, mode='user', number=5)
                if 'tweets' in tweets_data:
                    for tweet in tweets_data['tweets']:
                        tweet_text = tweet['text']
                        sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                        st.markdown(create_card(tweet_text, sentiment), unsafe_allow_html=True)
                else:
                    st.error("No tweets found or scraping failed.")
            else:
                st.warning("Enter a valid username.")

if __name__ == "__main__":
    main()
