from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import streamlit as st

# --- 1. DOWNLOAD THE WORD LIST (Safety Check) ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: # Use the general LookupError for missing resources
    nltk.download('vader_lexicon')


# --- 2. THE MACHINE FUNCTION (Slightly modified to RETURN data) ---
# This function does the analysis but sends the results back instead of printing them.
def get_sentiment_streamlit(text_to_analyze):
    """Analyzes text using VADER and returns the sentiment scores and mood."""
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text_to_analyze)
    compound_score = vs['compound']

    # Determine Mood based on the Compound Score
    if compound_score >= 0.05:
        mood = "Positive 😊"
    elif compound_score <= -0.05:
        mood = "Negative 😞"
    else:
        mood = "Neutral 😐"

    # Return all the values needed for Streamlit to display
    return mood, vs['neg'], vs['neu'], vs['pos'], compound_score


# --- 3. THE STREAMLIT WEB APP (The new interface) ---

st.set_page_config(layout="wide") # Use wide layout for better view
st.title("Basic Text Sentiment Analyzer")
st.markdown("---") # Visual separator

st.markdown("""
    This application uses the **VADER (Valence Aware Dictionary and sEntiment Reasoner)** model 
    to classify the mood of the text you enter as Positive, Negative, or Neutral.
""")

# Get user input through a web text box
user_input = st.text_area("Enter your review or text here:", height=150, placeholder="Example: The service was excellent, but the price was a bit high.")

# Check if the user has typed anything and clicked the button
if st.button('Analyze Sentiment') and user_input:
    
    # Call the analysis function
    mood, neg, neu, pos, compound = get_sentiment_streamlit(user_input)

    # --- Display Results ---
    st.subheader("Analysis Results:")
    
    # Display the final mood score
    st.metric(label="Overall Mood", value=mood, delta=f"Compound Score: {compound:.4f}")
    
    st.markdown("### Polarity Breakdown")
    col1, col2, col3 = st.columns(3)

    # Use columns to display individual scores cleanly
    with col1:
        st.metric(label="Positive Score", value=f"{pos*100:.1f}%")
        st.progress(pos, text="Positive")
    with col2:
        st.metric(label="Neutral Score", value=f"{neu*100:.1f}%")
        st.progress(neu, text="Neutral")
    with col3:
        st.metric(label="Negative Score", value=f"{neg*100:.1f}%")
        st.progress(neg, text="Negative")

    st.markdown("---")

    st.info(f"Your input was: **{user_input}**")
