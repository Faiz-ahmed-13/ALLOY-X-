# EXECUTED_CODE/10_alloyx_dashboard.py
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import os
from pathlib import Path
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ALLOY-X | AI Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with white text for specific elements
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: white !important;  /* Changed to white */
        border-left: 4px solid #667eea;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .result-defective {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    .result-nondefective {
        background: linear-gradient(135deg, #1dd1a1 0%, #10ac84 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-bar {
        height: 25px;
        background: #ecf0f1;
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #1dd1a1, #feca57, #ff6b6b);
        transition: width 1s ease-in-out;
    }
    
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        margin: 1rem 0;
        background: #f8f9fa;
        transition: all 0.3s ease;
        color: #2c3e50 !important;
    }
    
    .upload-box h3 {
        color: #2c3e50 !important;
        margin-bottom: 1rem;
    }
    
    .upload-box p {
        color: #5a6c7d !important;
        margin: 0.5rem 0;
    }
    
    .upload-box:hover {
        background: #e9ecef;
        border-color: #764ba2;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        color: #2c3e50 !important;  /* Fixed: Dark text on white background */
    }
    
    .feature-card h4 {
        color: #2c3e50 !important;  /* Fixed: Dark text for headings */
        margin-bottom: 0.5rem;
    }
    
    .feature-card p {
        color: #5a6c7d !important;  /* Fixed: Dark text for paragraphs */
    }
    
    /* White text for sidebar buttons */
    .stButton button {
        color: white !important;
        border: 1px solid #667eea !important;
        background-color: #667eea !important;
    }
    
    .stButton button:hover {
        color: white !important;
        background-color: #764ba2 !important;
        border-color: #764ba2 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_files = ['best_defect_detection_model.h5', 'final_defect_detection_model.h5']
        for model_path in model_files:
            if os.path.exists(model_path):
                return keras.models.load_model(model_path, compile=False)
        st.error("❌ No model found!")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to match model input size
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def create_confidence_gauge(confidence, is_defective):
    """Create a beautiful confidence gauge chart"""
    if is_defective:
        color = 'red'
        title = 'Defect Confidence'
    else:
        color = 'green'
        title = 'Quality Confidence'
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "lightgreen"} if not is_defective else 
                {'range': [50, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Header Section
    st.markdown('<h1 class="main-header">🔍 ALLOY-X AI Defect Detection System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; font-size: 1.2rem; color: #7f8c8d; margin-bottom: 2rem;'>
        Industrial-Grade Metal Surface Defect Detection powered by Deep Learning
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
        st.markdown("### 🚀 ALLOY-X Dashboard")
        st.markdown("---")
        
        # Model Information
        st.markdown("#### 📊 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "86.89%")
            st.metric("Recall", "100%")
        with col2:
            st.metric("Precision", "86.89%")
            st.metric("F1-Score", "92.98%")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("#### ⚡ Quick Actions")
        if st.button("🔄 Clear Session", use_container_width=True):
            st.rerun()
        
        if st.button("📊 View Model Details", use_container_width=True):
            st.session_state.show_model_details = True
        
        st.markdown("---")
        
        # System Info
        st.markdown("#### ℹ️ System Info")
        st.text(f"Model: CNN (4 Conv Layers)")
        st.text(f"Input Size: 224×224px")
        st.text(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Main Content - Two Columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📤 Upload Metal Surface Image</div>', unsafe_allow_html=True)
        
        # File Uploader with enhanced UI
        uploaded_file = st.file_uploader(
            " ",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a high-quality image of a metal surface for AI analysis",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display uploaded image with enhancements
            image = Image.open(uploaded_file)
            
            # Create two columns for image display
            img_col1, img_col2 = st.columns([2, 1])
            
            with img_col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with img_col2:
                # Image info
                st.markdown("**Image Details:**")
                st.text(f"Format: {uploaded_file.type}")
                st.text(f"Size: {image.size[0]}×{image.size[1]}")
                st.text(f"Mode: {image.mode}")
            
            # Analysis button with loading state
            if st.button("🔍 Analyze for Defects", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing the image..."):
                    # Simulate processing time for better UX
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Load model and make prediction
                    model = load_model()
                    
                    if model is not None:
                        # Preprocess image
                        processed_image = preprocess_image(image)
                        
                        # Make prediction
                        prediction = model.predict(processed_image, verbose=0)
                        confidence = prediction[0][0]
                        
                        # Determine result
                        is_defective = confidence > 0.5
                        result_class = "DEFECTIVE" if is_defective else "NON-DEFECTIVE"
                        confidence_percent = confidence * 100 if is_defective else (1 - confidence) * 100
                        
                        # Store results in session state
                        st.session_state.prediction_result = {
                            'is_defective': is_defective,
                            'confidence': confidence_percent,
                            'raw_confidence': confidence,
                            'image': image,
                            'timestamp': datetime.now()
                        }
        
        else:
            # Show upload prompt
            st.markdown("""
            <div class="upload-box">
                <h3>📁 Drag & Drop Image Here</h3>
                <p>Supported formats: JPG, JPEG, PNG, BMP</p>
                <p>Optimal size: 500×500 pixels or larger</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample images for quick testing
            st.markdown("### 🚀 Quick Test Images")
            sample_col1, sample_col2, sample_col3 = st.columns(3)
            
            # You can replace these with actual sample images from your dataset
            with sample_col1:
                if st.button("Test Defective Sample", use_container_width=True):
                    # Simulate a defective sample
                    st.session_state.prediction_result = {
                        'is_defective': True,
                        'confidence': 92.5,
                        'raw_confidence': 0.925,
                        'image': None,
                        'timestamp': datetime.now()
                    }
            
            with sample_col2:
                if st.button("Test Non-Defective Sample", use_container_width=True):
                    # Simulate a non-defective sample
                    st.session_state.prediction_result = {
                        'is_defective': False,
                        'confidence': 87.3,
                        'raw_confidence': 0.127,
                        'image': None,
                        'timestamp': datetime.now()
                    }
    
    with col2:
        st.markdown('<div class="sub-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            
            # Result Display
            if result['is_defective']:
                st.markdown(f"""
                <div class="result-defective">
                    <h2>🚨 DEFECT DETECTED</h2>
                    <h3>Confidence: {result['confidence']:.2f}%</h3>
                    <p>Surface contains potential defects requiring inspection</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-nondefective">
                    <h2>✅ QUALITY PASS</h2>
                    <h3>Confidence: {result['confidence']:.2f}%</h3>
                    <p>No defects detected - meets quality standards</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence Gauge
            st.plotly_chart(create_confidence_gauge(result['raw_confidence'], result['is_defective']), 
                          use_container_width=True)
            
            # Detailed Analysis
            st.markdown("### 📈 Detailed Analysis")
            
            # Confidence breakdown
            col_break1, col_break2 = st.columns(2)
            with col_break1:
                st.metric("Defect Probability", f"{result['raw_confidence']*100:.2f}%")
                st.metric("Quality Probability", f"{(1-result['raw_confidence'])*100:.2f}%")
            
            with col_break2:
                st.metric("Risk Level", "HIGH" if result['is_defective'] else "LOW")
                st.metric("Recommended Action", "INSPECT" if result['is_defective'] else "APPROVE")
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            if result['is_defective']:
                st.warning("""
                **Immediate Actions Required:**
                - 🛑 Isolate the product for manual inspection
                - 🔍 Perform detailed quality assessment
                - 📋 Document defect characteristics
                - ⚙️ Review production parameters
                - 📊 Update quality control records
                """)
            else:
                st.success("""
                **Quality Assurance Passed:**
                - ✅ Product meets quality standards
                - 🏭 Continue with production workflow
                - 📦 Ready for packaging/shipping
                - 🔄 Maintain current production parameters
                - 📈 Record successful inspection
                """)
            
            # Technical Details (expandable)
            with st.expander("🔧 Technical Details"):
                st.text(f"Raw Prediction Score: {result['raw_confidence']:.4f}")
                st.text(f"Classification Threshold: 0.5")
                st.text(f"Model: Custom CNN (86.89% accuracy)")
                st.text(f"Analysis Timestamp: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        else:
            # Placeholder before analysis
            st.markdown("""
            <div style='text-align: center; padding: 4rem; color: #7f8c8d;'>
                <h3>🔍 Waiting for Analysis</h3>
                <p>Upload an image and click "Analyze for Defects" to see results here</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Features Section (below main content)
    st.markdown("---")
    st.markdown('<div class="sub-header">✨ ALLOY-X Features</div>', unsafe_allow_html=True)
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 High Accuracy</h4>
            <p>86.89% test accuracy with 100% defect recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div class="feature-card">
            <h4>⚡ Real-time Processing</h4>
            <p>Fast analysis under 2 seconds per image</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🏭 Industrial Grade</h4>
            <p>Designed for manufacturing quality control</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        <p>ALLOY-X Defect Detection System | Built with TensorFlow & Streamlit | MJCET AIML</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()