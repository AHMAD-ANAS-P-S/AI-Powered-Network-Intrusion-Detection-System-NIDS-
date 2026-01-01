import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime


# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================


st.set_page_config(
    page_title="AI-Powered Network IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Session state initialization
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = {'Normal': 0, 'Anomaly': 0}


# ============================================================================
# DATA GENERATION & LOADING
# ============================================================================


def generate_simulated_network_data(n_samples=10000):
    """
    Generate simulated network traffic data with features:
    - Protocol: TCP (0), UDP (1), ICMP (2)
    - Packet Length: 64-1500 bytes
    - Packet Rate: 0-1000 packets/second
    - Byte Rate: 0-10MB/sec
    - Duration: 0-3600 seconds
    """
    np.random.seed(42)
    
    normal_data = np.random.normal(
        loc=[1, 500, 100, 5000000, 600],
        scale=[0.5, 200, 50, 2000000, 300],
        size=(int(n_samples * 0.8), 5)
    )
    
    attack_data = np.random.normal(
        loc=[0, 100, 500, 50000000, 3600],
        scale=[0.3, 50, 100, 10000000, 100],
        size=(int(n_samples * 0.2), 5)
    )
    
    X = np.vstack([normal_data, attack_data])
    y = np.hstack([np.zeros(int(n_samples * 0.8)), np.ones(int(n_samples * 0.2))])
    
    X = np.clip(X, 0, None)  # Ensure non-negative values
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    df = pd.DataFrame(X, columns=[
        'Protocol Type',
        'Packet Length (bytes)',
        'Packet Rate (pkt/sec)',
        'Byte Rate (bytes/sec)',
        'Flow Duration (sec)'
    ])
    df['Label'] = y.astype(int)
    
    return df


def load_data():
    """
    Load data from CSV file if available, otherwise use simulated data.
    For production, update to: df = pd.read_csv('path/to/dataset.csv')
    """
    # Check for real dataset
    dataset_files = ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
                     'network_traffic.csv',
                     'dataset.csv']
    
    for file in dataset_files:
        if os.path.exists(file):
            st.info(f"✅ Loading real dataset: {file}")
            df = pd.read_csv(file)
            return df
    
    # Fallback to simulation
    st.info("📊 Using simulated network traffic data (no real dataset found)")
    return generate_simulated_network_data()


# ============================================================================
# MODEL TRAINING
# ============================================================================


def train_model(X_train, y_train):
    """Train Random Forest Classifier for anomaly detection."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def preprocess_data(df):
    """Preprocess and prepare data for model training."""
    
    # Handle Label column - CIC-IDS2017 uses ' Label' with space
    label_column = None
    for col in df.columns:
        if 'label' in col.lower():
            label_column = col
            break
    
    if label_column is None:
        st.error("❌ No Label column found in dataset!")
        return None
    
    # Convert labels to binary (0 = normal, 1 = attack)
    y = (df[label_column] != 'BENIGN').astype(int)
    
    # Drop label column from features
    X = df.drop(label_column, axis=1)
    
    # Select numeric columns only
    X = X.select_dtypes(include=[np.number])
    
    # ===== FIX: Remove infinite and extreme values =====
    # Replace infinity with NaN first
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with 0
    X = X.fillna(0)
    
    # Remove columns with all zeros (useless features)
    X = X.loc[:, (X != 0).any(axis=0)]
    
    # Clip extreme values to reasonable range (prevents overflow)
    X = X.clip(-1e10, 1e10)
    
    # Drop rows where all values are zero (malformed rows)
    X = X.loc[(X != 0).any(axis=1)]
    
    st.info(f"📊 Using {X.shape[1]} features from {X.shape[0]} samples")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Double check for any remaining issues
    if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
        st.warning("⚠️ Data still contains NaN/Inf after scaling. Applying additional fixes...")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler, X.columns


# ============================================================================
# PREDICTION & DETECTION
# ============================================================================


def predict_traffic(model, scaler, features):
    """Predict if network traffic is normal or anomalous.
    
    Enhanced to detect suspicious patterns even with limited features.
    """
    # Get the number of features the scaler expects
    n_features_expected = scaler.n_features_in_
    
    # Create a feature vector with the right number of features
    full_features = np.zeros(n_features_expected)
    full_features[:len(features)] = features
    
    # ANOMALY DETECTION LOGIC - Pattern-based
    # Extract our 5 features
    protocol_type = features[0]
    packet_length = features[1]
    packet_rate = features[2]
    byte_rate = features[3]
    flow_duration = features[4]
    
    # Calculate suspicious indicators
    suspicious_score = 0
    
    # Check 1: Very short packets with high rate = flooding/DoS
    if packet_length < 100 and packet_rate > 500:
        suspicious_score += 3
    
    # Check 2: Extremely high byte rate = data exfiltration/DDoS
    if byte_rate > 5000000:  # > 5 MB/sec
        suspicious_score += 3
    
    # Check 3: Minimum packet length with maximum rate = SYN flood pattern
    if packet_length == 64 and packet_rate > 900:
        suspicious_score += 4
    
    # Check 4: Very high packet rate = flooding attack
    if packet_rate > 800:
        suspicious_score += 2
    
    # Check 5: Very short flow with high activity = one-off attack
    if flow_duration < 10 and (packet_rate > 500 or byte_rate > 3000000):
        suspicious_score += 3
    
    # Check 6: ICMP protocol with high rate = unusual pattern
    if protocol_type == 2 and packet_rate > 300:
        suspicious_score += 2
    
    # Check 7: Combination of small packets + huge byte rate = impossible pattern
    if packet_length < 200 and byte_rate > 8000000:
        suspicious_score += 4
    
    # Determine if anomaly based on suspicious score
    if suspicious_score >= 5:
        # Anomaly detected
        label = "🚨 ANOMALY DETECTED"
        confidence = min(95, 70 + (suspicious_score * 5))  # Cap at 95%
        prediction = 1
    else:
        # Use the ML model for borderline cases
        features_scaled = scaler.transform([full_features])
        prediction_ml = model.predict(features_scaled)[0]
        probability_ml = model.predict_proba(features_scaled)[0]
        
        if prediction_ml == 1 or max(probability_ml) < 0.7:
            # Model also thinks it might be anomaly
            label = "🚨 ANOMALY DETECTED"
            confidence = max(probability_ml) * 100
            prediction = 1
        else:
            label = "✅ NORMAL"
            confidence = max(probability_ml) * 100
            prediction = 0
    
    return label, confidence, prediction




# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_feature_importance(model, feature_names):
    """Visualize feature importance in model decisions."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Show only top 15 features (dataset has 68 features, too many to display)
    top_n = min(15, len(importances))
    top_indices = indices[:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(top_n), importances[top_indices], align='center', color='#3498db')
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in top_indices], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance in Intrusion Detection')
    plt.tight_layout()
    
    return fig



def plot_detection_stats():
    """Visualize detection statistics."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(st.session_state.detection_count.keys())
    values = list(st.session_state.detection_count.values())
    colors = ['#2ecc71', '#e74c3c']
    
    # Handle case where all values are 0 (no detections yet)
    if sum(values) == 0:
        ax.text(0.5, 0.5, 'No detections yet\nTest with Live Simulator →', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
    else:
        # Only show pie chart if there are detections
        ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    
    ax.set_title('Detection Statistics')
    
    return fig



def plot_traffic_distribution(df):
    """Visualize network traffic distribution."""
    
    # Find the label column (might have space or different name)
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    
    if label_col is None:
        # If no label column, just show distributions without separation
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Label column not found in raw data.\nShowing feature distributions instead.', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    # Get only the first 5 numeric columns for visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, feature in enumerate(numeric_cols):
        # Filter data based on label column
        normal_data = df[df[label_col] == 'BENIGN'][feature]
        anomaly_data = df[df[label_col] != 'BENIGN'][feature]
        
        axes[idx].hist(normal_data, bins=30, alpha=0.6, label='Normal', color='green')
        axes[idx].hist(anomaly_data, bins=30, alpha=0.6, label='Anomaly', color='red')
        axes[idx].set_title(f'Distribution: {feature}')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
    
    axes[-1].remove()  # Remove extra subplot
    plt.tight_layout()
    
    return fig



# ============================================================================
# MAIN UI - STREAMLIT APP
# ============================================================================


# Title & Description
st.title("🛡️ AI-Powered Network Intrusion Detection System")
st.markdown("""
    **Real-time anomaly detection using Machine Learning**
    
    This system uses a Random Forest classifier to identify suspicious network traffic patterns
    and alert administrators to potential intrusions.
""")


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration Panel")
    st.markdown("---")
    
    # Data Loading Section
    st.subheader("1️⃣ Load Data")
    if st.button("📥 Load Network Traffic Data", key="load_btn"):
        with st.spinner("Loading data..."):
            st.session_state.data = load_data()
        st.success(f"✅ Loaded {len(st.session_state.data)} samples")
    
    st.markdown("---")
    
    # Model Training Section
    st.subheader("2️⃣ Train Model")
    if st.button("🤖 Train Model Now", key="train_btn"):
        if 'data' not in st.session_state:
            st.error("⚠️ Please load data first!")
        else:
            with st.spinner("Training model (this may take a moment)..."):
                result = preprocess_data(st.session_state.data)
                
                if result is None:
                    st.error("❌ Error preprocessing data!")
                else:
                    X_train, X_test, y_train, y_test, scaler, feature_names = result
                    st.session_state.model = train_model(X_train, y_train)
                    st.session_state.scaler = scaler
                    st.session_state.feature_names = feature_names
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    
                    accuracy = st.session_state.model.score(X_test, y_test)
                    st.session_state.training_history.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'accuracy': accuracy
                    })
                    
                    st.success(f"✅ Model trained! Accuracy: {accuracy:.2%}")
    
    st.markdown("---")
    
    # System Status
    st.subheader("📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.model is not None:
            st.metric("Model Status", "✅ Ready")
        else:
            st.metric("Model Status", "⏳ Not Ready")
    with col2:
        st.metric("Detections Made", sum(st.session_state.detection_count.values()))


# Main Content Area
if st.session_state.model is None:
    st.warning("⚠️ **Steps to begin:**\n1. Load Network Traffic Data\n2. Train Model\n3. Test with Live Simulator")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Live Simulator", "📈 Analytics", "📋 Logs"])
    
    # TAB 1: Dashboard
    with tab1:
        st.subheader("System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Normal Traffic", st.session_state.detection_count['Normal'])
        with col2:
            st.metric("Anomalies Detected", st.session_state.detection_count['Anomaly'])
        with col3:
            total = sum(st.session_state.detection_count.values())
            if total > 0:
                anomaly_rate = (st.session_state.detection_count['Anomaly'] / total) * 100
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
        with col4:
            if 'X_test' in st.session_state and 'y_test' in st.session_state:
                accuracy = st.session_state.model.score(st.session_state.X_test, st.session_state.y_test)
                st.metric("Model Accuracy", f"{accuracy:.2%}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_detection_stats())
        with col2:
            st.pyplot(plot_feature_importance(st.session_state.model, st.session_state.feature_names))
    
    # TAB 2: Live Simulator
    with tab2:
        st.subheader("🔍 Live Traffic Simulator")
        st.markdown("Simulate network packets and test intrusion detection in real-time")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            protocol = st.selectbox("Protocol Type", ["TCP (0)", "UDP (1)", "ICMP (2)"])
            # Fixed: Correct parsing of protocol value
            protocol_val = float(protocol.split("(")[1].strip(")"))
        with col2:
            packet_length = st.slider("Packet Length (bytes)", 64, 1500, 512)
        with col3:
            packet_rate = st.slider("Packet Rate (pkt/sec)", 0, 1000, 100)
        
        col4, col5 = st.columns(2)
        with col4:
            byte_rate = st.slider("Byte Rate (bytes/sec)", 0, 10000000, 500000)
        with col5:
            flow_duration = st.slider("Flow Duration (sec)", 0, 3600, 600)
        
        if st.button("🚀 Analyze Packet", key="analyze_btn"):
            features = [protocol_val, packet_length, packet_rate, byte_rate, flow_duration]
            label, confidence, prediction = predict_traffic(st.session_state.model, st.session_state.scaler, features)
            
            st.session_state.detection_count['Normal' if prediction == 0 else 'Anomaly'] += 1
            
            if prediction == 0:
                st.success(f"{label} (Confidence: {confidence:.1f}%)")
            else:
                st.error(f"{label} (Confidence: {confidence:.1f}%)")
                st.warning("""
                **Recommended Actions:**
                - Block source IP address
                - Escalate to security team
                - Initiate incident response protocol
                """)
    
    # TAB 3: Analytics
    with tab3:
        st.subheader("📈 Detailed Analytics")
        
        if 'data' in st.session_state:
            st.pyplot(plot_traffic_distribution(st.session_state.data))
    
    # TAB 4: Logs
    with tab4:
        st.subheader("📋 Detection Logs")
        
        if st.session_state.training_history:
            history_df = pd.DataFrame(st.session_state.training_history)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No training history yet")


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <small>AI-Powered Network Intrusion Detection System | Powered by Scikit-Learn & Streamlit</small>
    </div>
""", unsafe_allow_html=True)