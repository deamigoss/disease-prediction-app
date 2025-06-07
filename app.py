import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy.sparse import issparse
from googletrans import Translator
from datetime import datetime
import pytz
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import psutil
import re
import string
import time
import joblib
import os
import gc

# =============================================
# Memory Management Configuration
# =============================================
MEMORY_THRESHOLD_MB = 500  # Memory limit in MB
INACTIVITY_TIMEOUT = 300   # 5 minutes in seconds
CLEAN_INTERVAL = 120        # 2 minutes in seconds

# Initialize session state for memory management
if 'last_activity' not in st.session_state:
    st.session_state.last_activity = time.time()
if 'last_clean' not in st.session_state:
    st.session_state.last_clean = time.time()

# =============================================
# Memory Management Functions
# =============================================
def update_activity():
    """Update last activity timestamp"""
    st.session_state.last_activity = time.time()

def memory_cleanup():
    """Perform comprehensive memory cleanup"""
    # Clear Streamlit caches
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # Clear TensorFlow/Keras sessions
    tf.keras.backend.clear_session()
    
    # Remove large objects from session state
    for key in list(st.session_state.keys()):
        if key in ['current_model', 'tfidf_vectorizer', 'translator', 'models']:
            del st.session_state[key]
    
    # Force garbage collection
    gc.collect()

def check_resources():
    """Check memory usage and inactivity"""
    current_time = time.time()
    mem_usage = psutil.Process().memory_info().rss / (1024 ** 2)
    
    # Scheduled cleanup every CLEAN_INTERVAL
    if current_time - st.session_state.last_clean > CLEAN_INTERVAL:
        with st.spinner('Optimizing memory...'):
            memory_cleanup()
            st.session_state.last_clean = current_time
            st.toast(f"🧹 Periodic memory cleanup (Usage: {mem_usage:.1f}MB)", icon="✅")
    
    # Emergency cleanup if memory threshold exceeded
    if mem_usage > MEMORY_THRESHOLD_MB:
        with st.spinner('Freeing up memory...'):
            memory_cleanup()
            st.toast(f"🚨 Emergency memory cleanup (Usage: {mem_usage:.1f}MB)", icon="⚠️")
            st.session_state.last_clean = current_time
    
    # Full reset after inactivity period
    if current_time - st.session_state.last_activity > INACTIVITY_TIMEOUT:
        memory_cleanup()
        st.rerun()

# =============================================
# Core Application Functions
# =============================================
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')

def preprocess_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    
    return ' '.join(tokens)

@st.cache_data(max_entries=1, ttl=3600)
def load_data():
    """Load and preprocess data in memory-efficient chunks"""
    data = pd.read_csv('dataset.csv', usecols=['text', 'label'])
    data = data.dropna(subset=['text', 'label'])
    
    # Process in smaller chunks
    chunk_size = 500
    processed_texts = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        processed_texts.extend(chunk['text'].apply(preprocess_text))
        gc.collect()
    
    data['processed_text'] = processed_texts
    return data

# =============================================
# Model Training Functions
# =============================================
def build_lightweight_ann(input_dim, output_dim):
    """Create optimized ANN model"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model_type, X_train, y_train, num_classes):
    """Memory-efficient model training"""
    if model_type == "ANN":
        model = build_lightweight_ann(X_train.shape[1], num_classes)
        model.fit(
            X_train, y_train,
            epochs=15,
            batch_size=8,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=2)],
            verbose=0
        )
    elif model_type == "SVM":
        model = SVC(kernel='linear', probability=True, cache_size=100)
        model.fit(X_train, y_train)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=1)
        model.fit(X_train, y_train)
    elif model_type == "Naive Bayes":
        model = MultinomialNB()
        model.fit(X_train, y_train)
    
    return model

# --- Setup Translator ---
translator = Translator()

@st.cache_data(ttl=3600, max_entries=1000)
def translate_to_english(text):
    """Translate Indonesian to English with error handling"""
    try:
        time.sleep(0.3)  # Prevent rate limiting
        result = translator.translate(text, src='id', dest='en')
        return result.text
    except Exception as e:
        st.error(f"Error menerjemahkan ke Inggris: {e}")
        return text  # Return original if translation fails

@st.cache_data(ttl=3600, max_entries=1000)
def translate_to_indonesian(text):
    """Translate English to Indonesian with error handling"""
    try:
        time.sleep(0.3)
        result = translator.translate(text, src='en', dest='id')
        return result.text
    except Exception as e:
        st.error(f"Error menerjemahkan ke Indonesia: {e}")
        return text
    

# =============================================
# Streamlit UI and Main Application
# =============================================
def main():
    # Initialize resources
    download_nltk_resources()
    check_resources()
    update_activity()
    
    # Load data
    data = load_data()
    X = data['processed_text']
    y = data['label']
    
    # Prepare models and vectorizer
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Model selection options
    model_options = {
        "Naive Bayes": "Multinomial Naive Bayes",
        "SVM": "Support Vector Machine",
        "Random Forest": "Random Forest",
        "ANN": "Neural Network"
    }
    
    # =============================================
    # UI Components
    # =============================================
    st.title("🩺 Sistem Prediksi Penyakit Berbasis Gejala")
    st.markdown("Aplikasi ini menggunakan teknik NLP untuk memprediksi penyakit berdasarkan deskripsi gejala.")
    
    # Sidebar
    with st.sidebar:
        st.header("Pengaturan Model")
        selected_model = st.selectbox("Pilih Model", list(model_options.keys()), index=0)
        
        # Memory monitor
        mem_usage = psutil.Process().memory_info().rss / (1024 ** 2)
        st.metric("Memory Usage", f"{mem_usage:.1f} MB")
        st.progress(
            min(1.0, mem_usage/MEMORY_THRESHOLD_MB),
            text=f"Memory usage: {mem_usage:.1f}/{MEMORY_THRESHOLD_MB}MB"
        )
        
        if st.button("🔄 Force Memory Cleanup"):
            memory_cleanup()
            st.rerun()

            st.markdown("---")
    st.markdown("**Informasi Dataset:**")
    st.write(f"Jumlah Data: {len(data)}")
    st.write(f"Jumlah Penyakit: {len(label_encoder.classes_)}")

    st.markdown("---")
    st.markdown("**Daftar Penyakit:**")

    # Terjemahkan label penyakit ke Bahasa Indonesia
    translated_classes = []
    for disease in label_encoder.classes_:
        try:
            translated = translate_to_indonesian(disease)
            translated_classes.append(translated)
        except:
            translated_classes.append(disease)
    
    # Buat dataframe dan tampilkan dalam sidebar
    df_labels_id = pd.DataFrame({'Penyakit': translated_classes})
    st.dataframe(df_labels_id, use_container_width=True)
    
    # Main content
    if selected_model != st.session_state.get('current_model_type'):
        with st.spinner(f'Preparing {selected_model} model...'):
            # Clean previous model
            if 'current_model' in st.session_state:
                if st.session_state.current_model_type == "ANN":
                    tf.keras.backend.clear_session()
                del st.session_state.current_model
                gc.collect()
            
            # Train new model
            try:
                train_data = X_train_tfidf.toarray() if selected_model == "ANN" else X_train_tfidf
                model = train_model(selected_model, train_data, y_train, len(label_encoder.classes_))
                
                # Store in session state
                st.session_state.current_model = model
                st.session_state.current_model_type = selected_model
                st.success(f"Model {selected_model} siap digunakan!")
            except Exception as e:
                st.error(f"Gagal melatih model: {str(e)}")
    
    # Prediction interface
    st.subheader("Masukkan Gejala Anda")
    user_input = st.text_area(
        "Deskripsikan gejala:",
        placeholder="Contoh: Saya mengalami demam tinggi, sakit kepala...",
        on_change=update_activity
    )
    
if st.button("Prediksi Penyakit", on_click=update_activity):
    if not user_input:
        st.warning("Silakan masukkan deskripsi gejala")
    elif 'current_model' not in st.session_state:
        st.error("Model belum siap")
    else:
        with st.spinner('Memproses...'):
            try:
                # Langkah 1: Terjemahkan ke Inggris
                with st.spinner('Menerjemahkan gejala...'):
                    input_en = translate_to_english(user_input)
                
                # Langkah 2: Preprocessing
                with st.spinner('Memproses teks...'):
                    processed_input = preprocess_text(input_en)
                    input_tfidf = tfidf_vectorizer.transform([processed_input])
                
                # Langkah 3: Prediksi
                with st.spinner('Menganalisis gejala...'):
                    if selected_model == "ANN":
                        pred = st.session_state.current_model.predict(input_tfidf.toarray())
                        pred_class = pred.argmax()
                        confidence = pred.max()
                    else:
                        pred = st.session_state.current_model.predict_proba(input_tfidf)
                        pred_class = pred.argmax()
                        confidence = pred.max()
                    
                    # Langkah 4: Terjemahkan hasil kembali ke Indonesia
                    disease_en = label_encoder.inverse_transform([pred_class])[0]
                    disease_id = translate_to_indonesian(disease_en)
                
                # Tampilkan hasil
                st.success("**Hasil Prediksi:**")
                st.markdown(f"**Penyakit:** {disease_id}")
                st.markdown(f"**Tingkat Kepercayaan:** {confidence*100:.1f}%")
                
                # Tampilkan 3 prediksi teratas (kecuali untuk SVM)
                if selected_model != "SVM":
                    st.markdown("**Kemungkinan Lain:**")
                    top_indices = np.argsort(pred[0])[-3:][::-1]  # Ambil top 3
                    
                    for idx in top_indices:
                        if idx != pred_class:  # Skip hasil utama
                            disease_en = label_encoder.inverse_transform([idx])[0]
                            disease_id = translate_to_indonesian(disease_en)
                            prob = pred[0][idx]
                            st.write(f"- {disease_id} ({prob*100:.1f}%)")
                            
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
            finally:
                memory_cleanup()

if __name__ == "__main__":
    main()

# ---Data Exploration Section ---
expander = st.expander("Eksplorasi Data")
with expander:
    tab1, tab2 = st.tabs(["Contoh Data", "Distribusi Penyakit"])
    
    with tab1:
        st.subheader("Contoh Data Latih")
        st.write(data[['text', 'label']].sample(10))  # Menggunakan sample() untuk contoh acak
    
    with tab2:
        st.subheader("Distribusi Penyakit dalam Dataset")
        
        # Hitung distribusi kelas
        class_dist = data['label'].value_counts().reset_index()
        class_dist.columns = ['Penyakit', 'Jumlah']
        
        # Ambil top 10 penyakit untuk efisiensi
        class_dist_top = class_dist.head(10).copy()
        
        # Terjemahkan nama penyakit ke Bahasa Indonesia
        class_dist_top['Penyakit'] = class_dist_top['Penyakit'].apply(
            lambda x: translate_to_indonesian(x) if pd.notnull(x) else x
        )
        
        # Tampilkan visualisasi
        st.bar_chart(class_dist_top.set_index('Penyakit'))
        st.dataframe(class_dist_top, use_container_width=True)