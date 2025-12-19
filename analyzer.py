import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="AI Emotion Analyzer Pro", layout="wide", page_icon="🧠")

# 2. Load the AI Model (Cached for performance)
@st.cache_resource
def load_model():
    # Using a specialized RoBERTa model for 7-way emotion classification
    return pipeline("text-classification", 
                    model="j-hartmann/emotion-english-distilroberta-base", 
                    return_all_scores=True)

classifier = load_model()

# 3. Initialize History in Session State
if 'history' not in st.session_state:
    st.session_state.history = []

# 4. Sidebar: History & About
with st.sidebar:
    st.title("Settings & History")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
    
    st.write("---")
    st.subheader("Previous Analyses")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            st.info(f"**{item['Emotion']}** ({item['Score']}%)\n\n*\"{item['Text'][:30]}...\"*")
    else:
        st.write("No history yet!")

# 5. Main UI
st.title("🧠 Advanced AI Emotion Analyzer")
st.markdown("Type any text below to see its detailed emotional breakdown.")

user_input = st.text_area("What are you thinking?", placeholder="e.g., I am so excited about graduating with my MCA!", height=150)

if st.button("Analyze Emotion", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("AI is thinking..."):
            # Get model results
            raw_results = classifier(user_input)[0]
            
            # Format results into a DataFrame
            df = pd.DataFrame(raw_results)
            df['score'] = (df['score'] * 100).round(2) # Convert to percentage
            
            # Find the top emotion
            top_row = df.loc[df['score'].idxmax()]
            
            # Add to history
            st.session_state.history.append({
                "Text": user_input,
                "Emotion": top_row['label'].title(),
                "Score": top_row['score']
            })

            # Display Results in Columns
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Primary Emotion")
                st.metric(label="Detected", value=top_row['label'].title(), delta=f"{top_row['score']}% Confidence")
                
                # Bar Chart
                fig_bar = px.bar(df, x='score', y='label', orientation='h', 
                                 title="Emotional Breakdown (%)",
                                 labels={'score': 'Confidence %', 'label': 'Emotion'},
                                 color='score', color_continuous_scale='RdPu')
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                st.subheader("Emotional DNA")
                # Radar Chart (Spider Chart)
                fig_radar = px.line_polar(df, r='score', theta='label', line_close=True)
                fig_radar.update_traces(fill='toself', line_color='#FF4B4B')
                st.plotly_chart(fig_radar, use_container_width=True)

st.write("---")
st.caption("Built for Megha Prasad | Powered by Hugging Face Transformers & Streamlit")


