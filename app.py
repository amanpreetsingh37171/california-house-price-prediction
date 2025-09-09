import streamlit as st
import os
import urllib.request
import subprocess, sys
import importlib
from huggingface_hub import hf_hub_download
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import json
import threading 
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import speech_recognition as sr
import queue
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import google.generativeai as genai
import webbrowser
# Configure Gemini (securely via Streamlit secrets)
genai.configure(api_key=st.secrets["GEMINI"])

# from main_old import rmse_results  # Import your RMSE dictionary

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="🏠 House Price Prediction",
    page_icon="💰",
    layout="wide"
)

# ---------------- DARK MODE TOGGLE ----------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

theme = st.sidebar.radio("🎨 Choose Theme", ["Dark", "Light"], index=0)
if theme == "Dark":
    st.markdown(
        """
        <style>
            .main { background: linear-gradient(to right, #0f172a, #1e293b); }
            h1, h2, h3, h4, h5, h6, p, .stMetric { color: #f8fafc !important; }
            .stButton>button { background: #9333ea; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
            .main { background: linear-gradient(to right, #dbeafe, #f8fafc); }
            h1, h2, h3, h4, h5, h6, p { color: #1e293b !important; }
            .stButton>button { background: #2563eb; color: white; }
            .stMetric { background: #e0f7fa; border-radius: 10px; padding: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )




# def ensure_package(pkg, version=None):
#     try:
#         return importlib.import_module(pkg)
#     except ImportError:
#         if version:
#             subprocess.check_call([sys.executable, "-m", "pip", "install", f"{pkg}=={version}"])
#         else:
#             subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
#         return importlib.import_module(pkg)

# # force install huggingface_hub
# huggingface_hub = ensure_package("huggingface_hub", "0.16.4")


# ---------------- CACHE MODEL LOADING ----------------




def create_compressed_models():
    """Create physically smaller model files - run once"""
    if os.path.exists("model_compressed.pkl") and os.path.exists("pipeline_compressed.pkl"):
        return  # Already compressed
    
    # Load original models
    if os.path.exists("model.pkl") and os.path.exists("pipeline.pkl"):
        model = joblib.load("model.pkl")
        pipeline = joblib.load("pipeline.pkl")
        
        # ⚠️ Removed: tree_.value precision reduction (not allowed in sklearn)

        # Save with maximum compression
        joblib.dump(model, "model_compressed.pkl", compress=9)
        joblib.dump(pipeline, "pipeline_compressed.pkl", compress=9)
        
        # Show size comparison
        original_size = (os.path.getsize("model.pkl") + os.path.getsize("pipeline.pkl")) / (1024*1024)
        compressed_size = (os.path.getsize("model_compressed.pkl") + os.path.getsize("pipeline_compressed.pkl")) / (1024*1024)
        st.success(f"✅ Models compressed: {original_size:.1f}MB → {compressed_size:.1f}MB")



# def create_compressed_models():
#     """Create physically smaller model files - run once"""
#     import numpy as np
    
#     if os.path.exists("model_compressed.pkl") and os.path.exists("pipeline_compressed.pkl"):
#         return  # Already compressed
    
#     # Load original models
#     if os.path.exists("model.pkl") and os.path.exists("pipeline.pkl"):
#         model = joblib.load("model.pkl")
#         pipeline = joblib.load("pipeline.pkl")
        
#         # Reduce precision to save space
#         if hasattr(model, 'estimators_'):
#             for estimator in model.estimators_:
#                 if hasattr(estimator, 'tree_'):
#                     estimator.tree_.value = estimator.tree_.value.astype(np.float32)
        
#         # Save with maximum compression
#         joblib.dump(model, "model_compressed.pkl", compress=9)
#         joblib.dump(pipeline, "pipeline_compressed.pkl", compress=9)
        
#         # Show size comparison
#         original_size = (os.path.getsize("model.pkl") + os.path.getsize("pipeline.pkl")) / (1024*1024)
#         compressed_size = (os.path.getsize("model_compressed.pkl") + os.path.getsize("pipeline_compressed.pkl")) / (1024*1024)
#         st.success(f"✅ Models compressed: {original_size:.1f}MB → {compressed_size:.1f}MB")


# # Google Drive direct download links (replace 'id' values with your file IDs)
# model_url = "https://drive.google.com/uc?export=download&id=1cUe2WADBh9-QeGmKsgl9xJpBtKWS93vS"
# pipeline_url = "https://drive.google.com/uc?export=download&id=1TvWSbniMF3vhlR78qIKlWvk5fzw3BRWa"

# # Download if not already present
# if not os.path.exists("model.pkl"):
#     urllib.request.urlretrieve(model_url, "model.pkl")

# if not os.path.exists("pipeline.pkl"):
#     urllib.request.urlretrieve(pipeline_url, "pipeline.pkl")

# # Load with joblib
# model = joblib.load("model.pkl")
# pipeline = joblib.load("pipeline.pkl")


# Create compressed models first
create_compressed_models()

# Load compressed models
if os.path.exists("model_compressed.pkl") and os.path.exists("pipeline_compressed.pkl"):
    model = joblib.load("model_compressed.pkl")
    pipeline = joblib.load("pipeline_compressed.pkl")
else:
    # Fallback to original Google Drive download
    model_url = "https://drive.google.com/uc?export=download&id=1cUe2WADBh9-QeGmKsgl9xJpBtKWS93vS"
    pipeline_url = "https://drive.google.com/uc?export=download&id=1TvWSbniMF3vhlR78qIKlWvk5fzw3BRWa"
    
    if not os.path.exists("model.pkl"):
        urllib.request.urlretrieve(model_url, "model.pkl")
    if not os.path.exists("pipeline.pkl"):
        urllib.request.urlretrieve(pipeline_url, "pipeline.pkl")
    
    create_compressed_models()  # Compress after download
    model = joblib.load("model_compressed.pkl")
    pipeline = joblib.load("pipeline_compressed.pkl")


# ---------------- LOAD MODEL FROM HUGGING FACE ----------------


# @st.cache_data(show_spinner=True)
# def load_models():
#     """Load models from HuggingFace with error handling"""
#     try:
#         with st.spinner("🔄 Loading models from HuggingFace..."):
#             # Check if secrets are available
#             if "HUGGING_FACE" in st.secrets:
#                 token = st.secrets["HUGGING_FACE"]["token"]
#             else:
#                 token = None  # For public repos
            
#             model_path = hf_hub_download(
#                 repo_id="Amanpreet3023/california-house-price-model",
#                 filename="model.pkl",
#                 token=token
#             )
#             pipeline_path = hf_hub_download(
#                 repo_id="Amanpreet3023/california-house-price-pipeline",
#                 filename="pipeline.pkl",
#                 token=token
#             )
            
#             model = joblib.load(model_path)
#             pipeline = joblib.load(pipeline_path)
            
#             return model, pipeline
#     except Exception as e:
#         st.error(f"❌ Failed to load models: {str(e)}")
#         st.error("Please check if your HuggingFace repositories are public or if your token is correct.")
#         st.stop()

# model, pipeline = load_models()
# st.write("Models loaded successfully!")


# ---------------- LOAD MODEL ----------------
# model = joblib.load("model.pkl")
# pipeline = joblib.load("pipeline.pkl")

# ---------------- HEADER ----------------
st.title("🏠 California House Price Prediction")
st.markdown("""
Welcome to the **AI-Powered California House Price Estimator** 🚀  
This app predicts the estimated **median house value** based on important factors like  
🗺 **location**, 👨‍👩‍👧‍👦 **population**, 🏡 **rooms**, and 🌊 **ocean proximity**.  

👉 Use the sidebar to input house details and click **Predict**.
""")
st.markdown("---")


# Add this after st.markdown("---") 
if os.path.exists("model_compressed.pkl") and os.path.exists("pipeline_compressed.pkl"):
    model_size = os.path.getsize("model_compressed.pkl") / (1024*1024)
    pipeline_size = os.path.getsize("pipeline_compressed.pkl") / (1024*1024)
    total_size = model_size + pipeline_size
    st.info(f"📦 Compressed models loaded: {total_size:.1f}MB")


# ---------------- SIDEBAR ----------------
st.sidebar.header("🔧 Input Features")
st.sidebar.info("Fill in the house details below:")

longitude = st.sidebar.number_input("📍 Longitude", value=-122.23, format="%.2f", help="Geographical longitude of the house")
latitude = st.sidebar.number_input("📍 Latitude", value=37.88, format="%.2f", help="Geographical latitude of the house")
housing_median_age = st.sidebar.slider("🏘 Housing Median Age", 1, 100, 41, help="Median age of houses in the area")
total_rooms = st.sidebar.number_input("🚪 Total Rooms", value=880, help="Total number of rooms in the house block")
total_bedrooms = st.sidebar.number_input("🛏 Total Bedrooms", value=129, help="Number of bedrooms available")
population = st.sidebar.number_input("👨‍👩‍👧 Population", value=322, help="Population in the block area")
households = st.sidebar.number_input("🏡 Households", value=126, help="Number of households in the block")
median_income = st.sidebar.number_input("💵 Median Income (10k USD)", value=8.3252, format="%.4f", help="Median income of households in 10,000s")

ocean_proximity = st.sidebar.selectbox(
    "🌊 Ocean Proximity",
    ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"],
    help="Proximity of the house to the ocean"
)

# Autofill example
if st.sidebar.button("🎲 Use Example Data"):
    longitude, latitude = -118.30, 34.05
    housing_median_age, total_rooms, total_bedrooms = 25, 2000, 500
    population, households, median_income = 800, 300, 6.5
    ocean_proximity = "NEAR OCEAN"

# ---------------- INPUT DATA ----------------
input_data = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
}])

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📋 Input Data", "📊 Insights", "📑 Reports"])

# --- TAB 1: Input Data ---
with tab1:
    st.subheader("📋 Input Data Section")

    # Create three sub-tabs inside Input Data
    subtab1, subtab2, subtab3 = st.tabs(["🏡 Prediction", "🤖 Chatbot", "🧠 NLP Tasks"])


    # --- Sub-tab 1: Prediction ---
    with subtab1:
        st.subheader("📋 Input Features Preview")
        st.dataframe(input_data, use_container_width=True)

        if st.button("🚀 Predict House Price"):
            transformed_data = pipeline.transform(input_data)
            prediction = model.predict(transformed_data)
            price = prediction[0]

            st.success("✅ Prediction Complete!")

            # Display result card
            st.metric(label="💰 Predicted House Price", value=f"${price:,.2f}")
            # Convert to Indian Rupees (example rate: 1 USD = 83 INR)
            usd_to_inr = 83
            price_inr = price * usd_to_inr

            # Display in INR
            st.metric(label="💴 Predicted House Price (INR)", value=f"₹{price_inr:,.2f}")


            # Interpretation
            if price < 100000:
                st.info("🏚 Very Affordable Housing Area")
            elif 100000 <= price < 300000:
                st.info("🏠 Standard Mid-Range Housing")
            else:
                st.info("🏡 Premium & Expensive Location")

                st.session_state["prediction"] = price 



    with subtab2:
        if "chat_data" not in st.session_state:
            st.session_state.chat_data = []

        # Load API key from secrets.toml
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GEMINI"]["api_key"])

        model = genai.GenerativeModel("gemini-2.5-flash")

        st.header("🤖 Project Chatbot")
        st.subheader("Ask me anything!")

        # User input
        user_input = st.text_input("💬 Type your question here:")

        # Button to submit query
        if st.button("🚀 Ask"):
            if user_input:
                st.session_state.chat_data.append(("User", user_input))
            
                try:
                    # Call Gemini AI
                    response = model.generate_content(user_input)
                    st.session_state.chat_data.append(("AI", response.text))
                except Exception as e:
                 st.session_state.chat_data.append(("AI", f"❌ Error: {str(e)}"))

    # Display chat history
    for sender, message in st.session_state.chat_data:
        with st.chat_message(sender):
            st.markdown(message)

    

    with subtab3:
        st.subheader("🧠 NLP Tasks")

        if "nlp_history" not in st.session_state:
            st.session_state.nlp_history = []

        # Input box
        nlp_input = st.text_area("Enter text for NLP task:")

        task_type = st.selectbox("Choose NLP Task:", ["Summarization", "Sentiment Analysis", "Keyword Extraction"])

        if st.button("🚀 Run NLP Task"):
            if nlp_input:
                try:
                    # Call Gemini model
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    prompt = f"Perform {task_type} on the following text:\n{nlp_input}"
                    response = model.generate_content(prompt)

                    # Save history
                    st.session_state.nlp_history.append(("Input", nlp_input))
                    st.session_state.nlp_history.append(("Output", response.text))

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        # Display NLP history
        if st.session_state.nlp_history:
            for role, text in st.session_state.nlp_history:
                if role == "Input":
                    st.markdown(f"**📝 Input:** {text}")
                else:
                    st.markdown(f"**💡 Output:** {text}")



        # # ----------------------------
        # # RMSE Table & Approx. Accuracy

        # USD_TO_INR = 83  # Conversion rate USD → INR
        # AVERAGE_PRICE_USD = 200000  # For approximate accuracy %

        # # Convert dictionary to DataFrame
        # rmse_df = pd.DataFrame(rmse_results.items(), columns=["Model", "RMSE (USD)"])
        # rmse_df["RMSE (INR)"] = rmse_df["RMSE (USD)"] * USD_TO_INR
        # rmse_df["Approx. Accuracy (%)"] = (1 - rmse_df["RMSE (USD)"] / AVERAGE_PRICE_USD) * 100

        # # ----------------------------
        # # Display Table
        # st.subheader("📊 Model RMSE & Approximate Accuracy")
        # st.dataframe(rmse_df, use_container_width=True)
        # st.info("Lower RMSE indicates better predictions. Approx. Accuracy gives an intuitive sense of model performance.")

        # # ----------------------------
        # # Display Bar Chart
        # st.subheader("📊 RMSE Comparison Bar Chart")
        # fig, ax = plt.subplots(figsize=(8, 4))
        # sns.barplot(x="RMSE (USD)", y="Model", data=rmse_df, palette="viridis", ax=ax)
        # ax.set_title("Model RMSE Comparison")
        # ax.set_xlabel("RMSE (USD)")
        # ax.set_ylabel("Model")
        # st.pyplot(fig)



# --- TAB 2: Insights ---
with tab2:


    st.subheader("📊 Feature Insights & Interactive Visualizations")

    # Create a DataFrame for insights including input features + predicted price
    if "prediction" in st.session_state:
        house_data = input_data.copy()
        house_data["PredictedPrice"] = st.session_state["prediction"]
    else:
        house_data = input_data.copy()
        house_data["PredictedPrice"] = 0  # placeholder if no prediction yet

    # ------------------- Tabs inside Insights -------------------
    insights_tab1, insights_tab2, insights_tab3, insights_tab4, insights_tab5 = st.tabs([
        "🏠 Price by Bedrooms", 
        "📈 Feature vs Price", 
        "📊 Price Distribution", 
        "🗺 Map View",
        "🌌 3D Feature Analysis"
    ])

    # ---------------- Tab 1: Bar chart ----------------
    with insights_tab1:
        st.write("Predicted House Price by Number of Bedrooms")
        fig_bar = px.bar(
            house_data, 
            x='total_bedrooms', 
            y='PredictedPrice', 
            color='PredictedPrice',
            text='PredictedPrice',
            title='Predicted House Price by Bedrooms',
            labels={'total_bedrooms':'Number of Bedrooms', 'PredictedPrice':'Price in $'},
        
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_bar.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', title_font_size=20)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------- Tab 2: Scatter plot ----------------
    with insights_tab2:
        st.write("Feature Correlation with Predicted Price")
        fig_scatter = px.scatter(
            house_data, 
            x='total_rooms', 
            y='PredictedPrice', 
            color='total_bedrooms',
            size='population',
            hover_data=['households', 'median_income', 'ocean_proximity'],
            title='Rooms & Population vs Predicted Price',
            labels={'total_rooms':'Total Rooms', 'PredictedPrice':'Predicted Price'}
        )
        fig_scatter.update_layout(title_font_size=20)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ---------------- Tab 3: Histogram ----------------
    with insights_tab3:
        st.write("Distribution of Predicted Prices")
        fig_hist = px.histogram(
            house_data,
            x='PredictedPrice',
            nbins=10,
            color='total_bedrooms',
            marginal='box',  # adds boxplot on top
            title='Distribution of Predicted House Prices',

        )
        fig_hist.update_layout(title_font_size=20)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ---------------- Tab 4: Map ----------------
    with insights_tab4:
        if 'latitude' in house_data.columns and 'longitude' in house_data.columns:
            st.write("Predicted Prices on Map")
            fig_map = px.scatter_mapbox(
                house_data,
                lat='latitude',
                lon='longitude',
                color='PredictedPrice',
                size='PredictedPrice',
                hover_name='ocean_proximity',
                color_continuous_scale=px.colors.cyclical.IceFire,
                size_max=25,
                zoom=10,
                height=500,
                title='Geographical Distribution of Predicted Prices'
            )
            fig_map.update_layout(mapbox_style='open-street-map', title_font_size=20)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Latitude and Longitude data not available for map.")

    # ---------------- Tab 5: 3D Scatter ----------------
    with insights_tab5:
        st.write("🌌 3D Interactive Analysis of House Features")
        fig_3d = px.scatter_3d(
            house_data,
            x='total_rooms',
            y='total_bedrooms',
            z='PredictedPrice',
            color='median_income',
            size='population',
            hover_name='ocean_proximity',
            size_max=30,
            title='3D Analysis: Rooms vs Bedrooms vs Price'
        )
        fig_3d.update_layout(title_font_size=20, scene=dict(
            xaxis_title='Total Rooms',
            yaxis_title='Total Bedrooms',
            zaxis_title='Predicted Price ($)'
        ))
        st.plotly_chart(fig_3d, use_container_width=True)



# --- TAB 3: Reports ---
with tab3:
    st.subheader("📑 Download Prediction Report")

    if "prediction" in st.session_state:
        report = input_data.copy()
        report["Predicted Price"] = st.session_state["prediction"]

        # CSV Export
        csv_buffer = BytesIO()
        report.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Report as CSV",
            data=csv_buffer.getvalue(),
            file_name="house_price_report.csv",
            mime="text/csv"
        )

        # PDF Export
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("🏠 House Price Prediction Report", styles['Title']),
            Spacer(1, 12),
            Paragraph(f"📍 Location: ({longitude}, {latitude})", styles['Normal']),
            Paragraph(f"🏘 Median Age: {housing_median_age}", styles['Normal']),
            Paragraph(f"🚪 Total Rooms: {total_rooms}", styles['Normal']),
            Paragraph(f"🛏 Bedrooms: {total_bedrooms}", styles['Normal']),
            Paragraph(f"👨‍👩‍ Population: {population}", styles['Normal']),
            Paragraph(f"🏡 Households: {households}", styles['Normal']),
            Paragraph(f"💵 Median Income: {median_income}", styles['Normal']),
            Paragraph(f"🌊 Ocean Proximity: {ocean_proximity}", styles['Normal']),
            Spacer(1, 12),
            Paragraph(f"💰 Predicted Price: ${st.session_state['prediction']:,.2f} / ₹{st.session_state['prediction']*83:,.2f}", styles['Heading2'])

        ]
        doc.build(story)
        pdf_buffer.seek(0)

        st.download_button(
            label="📥 Download Report as PDF",
            data=pdf_buffer,
            file_name="house_price_report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("⚠️ Please run a prediction first.")
