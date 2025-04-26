import streamlit as st
import pickle

# Load the model and vectorizer
model = pickle.load(open('finalized_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))

# Streamlit App Title
st.title("📰 Fake News Detector")

# Input box for user
news_text = st.text_area("Enter the News Article Text:")

# When button is clicked
if st.button("Detect"):
    if news_text.strip() == "":
        st.warning("Please enter some text to predict!")
    else:
        # Transform the input and make prediction
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)

        # Show result
        if prediction[0] == 0:  # Assuming 1 = FAKE
            st.error("🚨 FAKE NEWS DETECTED!")
        else:
            st.success("✅ This news appears to be REAL.")

# Sidebar Info
st.sidebar.title("About")
st.sidebar.info("Built with ❤️ by Anant  Sharma")