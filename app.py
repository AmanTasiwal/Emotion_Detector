import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the trained model
MODEL_PATH = "svm_pipeline.joblib"  
ENCODER_PATH = "label_encoder.joblib"  # Label Encoder Path

pipe_svm = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)  # Load Label Encoder

# Mapping emotion labels to emojis
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

# Prediction function
def predict_emotions(docx):
    result = pipe_svm.predict([docx])[0]  # Model predicts a number
    predicted_label = label_encoder.inverse_transform([result])[0]  # Convert to text label
    return predicted_label

# Get prediction probabilities
def get_prediction_proba(docx):
    probabilities = pipe_svm.predict_proba([docx])
    return probabilities.tolist()  # Convert to list for Streamlit

# Streamlit UI
def main():
    st.set_page_config(page_title="Text Emotion Detection", page_icon="🎭")
    st.title("🎭 Text Emotion Detection App")
    st.write("Detect the emotion behind the text you enter.")

    # Sidebar for model details
    with st.sidebar:
        st.subheader("Model Info")
        st.write("🔹 **Algorithm**: SVM (Calibrated) with CountVectorizer")
        st.write("🔹 **Features**: Bigrams, Stop-word removal")
        st.write("🔹 **Dataset**: Emotion Dataset")

    # Input form
    with st.form(key='emotion_form'):
        raw_text = st.text_area("Type your text here:", height=150)
        submit_text = st.form_submit_button(label="Analyze Emotion")

    # Processing after submission
    if submit_text:
        col1, col2 = st.columns(2)

        # Make predictions
        prediction_label = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        # Left column: Original text & prediction
        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction_label, "❓")
            st.write(f"**Emotion**: {prediction_label} {emoji_icon}")
            st.write(f"**Confidence**: {np.max(probability):.4f}")

        # Right column: Probability visualization
        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_svm.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Emotion", "Probability"]

            # Probability bar chart
            chart = alt.Chart(proba_df_clean).mark_bar().encode(
                x=alt.X("Emotion", sort="-y"),
                y="Probability",
                color="Emotion"
            ).properties(width=350)

            st.altair_chart(chart, use_container_width=True)

if __name__ == '__main__':
    main()
