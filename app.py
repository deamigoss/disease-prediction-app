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
from googletrans import Translator
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
from memory_profiler import profile

# Constants
CLEAR_CACHE_EVERY = 15  # Clear cache every 15 interactions
MEMORY_THRESHOLD_MB = 800  # Clear cache if memory exceeds 800MB

# Initialize session state for cache management
if 'interaction_count' not in st.session_state:
    st.session_state.interaction_count = 0
if 'last_cache_clear' not in st.session_state:
    st.session_state.last_cache_clear = time.time()

# Cache management function
def manage_cache():
    """Intelligently manage cache based on interactions and memory usage"""
    st.session_state.interaction_count += 1
    
    # Check memory usage
    mem = psutil.Process().memory_info().rss / (1024 ** 2)
    mem_threshold = MEMORY_THRESHOLD_MB
    
    # Check if we should clear cache
    if (st.session_state.interaction_count % CLEAR_CACHE_EVERY == 0) or (mem > mem_threshold):
        # Clear various caches
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Clear TensorFlow session if it exists
        if 'models' in st.session_state and "ANN" in st.session_state.models:
            tf.keras.backend.clear_session()
        
        # Force garbage collection
        gc.collect()
        
        # Update tracking variables
        st.session_state.last_cache_clear = time.time()
        
        # Show notification
        reason = "memory threshold" if mem > mem_threshold else "periodic schedule"
        st.toast(f"🧹 Cache cleared ({reason}) - Memory: {mem:.1f}MB", icon="✅")

# --- Text Preprocessing ---
def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenization and stemming
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# --- NLTK Resource Download ---
@st.cache_resource(ttl=3600)  # Refresh every hour
def download_nltk_resources():
    """Download required NLTK data"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {e}")

download_nltk_resources()

# --- Data Loading ---
@st.cache_data(max_entries=1, ttl=3600)  # Keep for 1 hour
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    try:
        data = pd.read_csv('dataset.csv')
        data = data.drop(columns=['Unnamed: 0'], errors='ignore')
        
        # Preprocess the text data
        data['processed_text'] = data['text'].apply(preprocess_text)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

data = load_and_preprocess_data()

# Only proceed if data is loaded successfully
if not data.empty:
    # Split data
    X = data['processed_text']
    y = data['label']

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # --- TF-IDF Vectorizer ---
    @st.cache_resource
    def get_vectorizer():
        return TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    tfidf_vectorizer = get_vectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

# --- Model Selection ---
model_options = {
    "ANN": "Neural Network dengan Embedding",
    "SVM": "Support Vector Machine",
    "Random Forest": "Random Forest",
    "Naive Bayes": "Multinomial Naive Bayes"
}

# --- Translation Utilities ---
@st.cache_resource(ttl=3600)  # Cache translator for 1 hour
def get_translator():
    """Get cached translator instance"""
    return Translator()

translator = get_translator()

def translate_to_english(text):
    """Translate text to English with error handling"""
    if not text or not isinstance(text, str):
        return text
        
    try:
        time.sleep(0.3)  # Prevent rate limiting
        result = translator.translate(text, src='id', dest='en')
        return result.text
    except Exception as e:
        st.error(f"Error translating to English: {e}")
        return text

def translate_to_indonesian(text):
    """Translate text to Indonesian with error handling"""
    if not text or not isinstance(text, str):
        return text
        
    try:
        time.sleep(0.3)
        result = translator.translate(text, src='en', dest='id')
        return result.text
    except Exception as e:
        st.error(f"Error translating to Indonesian: {e}")
        return text

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Sistem Prediksi Penyakit Berbasis Gejala",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Sistem Prediksi Penyakit Berbasis Gejala")
st.markdown("""
Aplikasi ini menggunakan teknik NLP untuk memprediksi penyakit berdasarkan deskripsi gejala yang Anda berikan.
""")

# Sidebar for model selection and info
with st.sidebar:
    st.header("Pengaturan Model")
    
    selected_model = st.selectbox(
        "Pilih Model",
        list(model_options.keys()),
        index=0
    )

    st.markdown("---")
    st.markdown("**Informasi Dataset:**")
    st.write(f"Jumlah Data: {len(data)}")
    st.write(f"Jumlah Penyakit: {len(label_encoder.classes_)}")
    
    # Memory usage display
    mem = psutil.Process().memory_info().rss / (1024 ** 2)
    st.metric("Memory Usage", f"{mem:.1f} MB")
    st.progress(min(mem / MEMORY_THRESHOLD_MB, 1.0))
    
    st.markdown("---")
    st.markdown("**Daftar Penyakit:**")
    
    # Cached translation of disease labels
    @st.cache_data(ttl=3600)
    def get_translated_diseases(classes):
        translated = []
        for disease in classes:
            try:
                translated.append(translate_to_indonesian(disease))
            except:
                translated.append(disease)
        return translated
    
    translated_classes = get_translated_diseases(label_encoder.classes_)
    df_labels_id = pd.DataFrame({'Penyakit': translated_classes})
    st.dataframe(df_labels_id, use_container_width=True)
    
    # Cache info and management
    st.markdown("---")
    st.markdown("**Manajemen Cache**")
    st.write(f"Interaksi sejak clear terakhir: {st.session_state.interaction_count}")
    st.write(f"Terakhir dibersihkan: {time.ctime(st.session_state.last_cache_clear)}")
    
    if st.button("Bersihkan Cache Manual"):
        st.cache_data.clear()
        st.cache_resource.clear()
        if 'models' in st.session_state and "ANN" in st.session_state.models:
            tf.keras.backend.clear_session()
        gc.collect()
        st.session_state.last_cache_clear = time.time()
        st.session_state.interaction_count = 0
        st.toast("Cache dibersihkan secara manual", icon="🧹")

# Initialize session state for models
if 'models' not in st.session_state:
    st.session_state.models = {}
    st.session_state.current_model = None

# Train selected model only if not already trained
if selected_model not in st.session_state.models and not data.empty:
    if selected_model == "ANN":
        with st.spinner('Melatih model ANN...'):
            try:
                model = Sequential([
                    Dense(256, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
                    Dropout(0.5),
                    Dense(128, activation='relu'),
                    Dropout(0.3),
                    Dense(len(label_encoder.classes_), activation='softmax')
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
                
                history = model.fit(
                    X_train_tfidf.toarray(), y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                st.session_state.models["ANN"] = model
                st.session_state.current_model = "ANN"
                st.success("Model ANN berhasil dilatih!")
            except Exception as e:
                st.error(f"Gagal melatih model ANN: {str(e)}")
    
    elif selected_model == "SVM":
        with st.spinner('Melatih model SVM...'):
            try:
                model = SVC(kernel='linear', probability=True, random_state=42)
                model.fit(X_train_tfidf, y_train)
                st.session_state.models["SVM"] = model
                st.session_state.current_model = "SVM"
                st.success("Model SVM berhasil dilatih!")
            except Exception as e:
                st.error(f"Gagal melatih model SVM: {str(e)}")
    
    elif selected_model == "Random Forest":
        with st.spinner('Melatih model Random Forest...'):
            try:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_tfidf, y_train)
                st.session_state.models["Random Forest"] = model
                st.session_state.current_model = "Random Forest"
                st.success("Model Random Forest berhasil dilatih!")
            except Exception as e:
                st.error(f"Gagal melatih model Random Forest: {str(e)}")
    
    elif selected_model == "Naive Bayes":
        with st.spinner('Melatih model Naive Bayes...'):
            try:
                model = MultinomialNB()
                model.fit(X_train_tfidf, y_train)
                st.session_state.models["Naive Bayes"] = model
                st.session_state.current_model = "Naive Bayes"
                st.success("Model Naive Bayes berhasil dilatih!")
            except Exception as e:
                st.error(f"Gagal melatih model Naive Bayes: {str(e)}")

    # Update cache management after model training
    manage_cache()

# Get the current model
current_model = st.session_state.models.get(selected_model) if not data.empty else None

# Display model performance only if we have a trained model
if current_model is not None and not data.empty:
    st.subheader("Evaluasi Model")
    
    # Cached model evaluation
    @st.cache_data(ttl=600)  # Cache evaluation for 10 minutes
    def evaluate_model(_model, model_type, _X_test, _y_test):
        if model_type == "ANN":
            y_pred_probs = _model.predict(_X_test.toarray())
            y_pred = np.argmax(y_pred_probs, axis=1)
        else:
            y_pred = _model.predict(_X_test)
        
        accuracy = accuracy_score(_y_test, y_pred)
        report = classification_report(_y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        return accuracy, report
    
    accuracy, report = evaluate_model(current_model, selected_model, X_test_tfidf, y_test)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Akurasi pada Data Test", f"{accuracy:.2%}")
    
    with col2:
        st.metric("Jumlah Penyakit", len(label_encoder.classes_))
    
    # Show detailed classification report
    with st.expander("Lihat Detail Evaluasi"):
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))

# User input section
st.subheader("Masukkan Gejala Anda")
user_input = st.text_area(
    "Deskripsikan gejala yang Anda alami dalam bahasa Indonesia:",
    placeholder="Contoh: Saya mengalami demam tinggi, sakit kepala, dan nyeri otot...",
    height=150
)

prediction_col, _ = st.columns([0.5, 0.5])

with prediction_col:
    if st.button("Prediksi Penyakit", use_container_width=True):
        manage_cache()  # Track this interaction
        
        if not user_input:
            st.warning("Silakan masukkan deskripsi gejala terlebih dahulu.")
        elif current_model is None:
            st.error("Model belum dilatih. Silakan tunggu hingga proses training selesai.")
        else:
            with st.spinner('Memproses gejala dan membuat prediksi...'):
                try:
                    # Sub-spinner untuk translate
                    with st.spinner('Menerjemahkan gejala...'):
                        input_en = translate_to_english(user_input)
                    
                    # Sub-spinner untuk preprocessing
                    with st.spinner('Memproses teks...'):
                        processed_input = preprocess_text(input_en)
                        input_tfidf = tfidf_vectorizer.transform([processed_input])
                    
                    # Sub-spinner untuk prediksi
                    with st.spinner('Menganalisis...'):
                        if selected_model == "ANN":
                            pred_probs = current_model.predict(input_tfidf.toarray())
                            pred_class_idx = np.argmax(pred_probs, axis=1)[0]
                            confidence = np.max(pred_probs)
                        else:
                            pred_probs = current_model.predict_proba(input_tfidf)
                            pred_class_idx = current_model.predict(input_tfidf)[0]
                            confidence = np.max(pred_probs)
                        
                        # Get predicted class
                        pred_class_en = label_encoder.inverse_transform([pred_class_idx])[0]
                        pred_class_id = translate_to_indonesian(pred_class_en)
                    
                    # Display results
                    st.success("**Hasil Prediksi:**")
                    
                    result_container = st.container(border=True)
                    with result_container:
                        st.markdown(f"**Penyakit:** {pred_class_id}")
                        st.markdown(f"**Tingkat Kepercayaan:** {confidence*100:.1f}%")
                        
                        # Visual confidence indicator
                        st.progress(float(confidence))
                        
                        # Show top 3 predictions if available
                        if selected_model != "SVM":  # SVM's predict_proba can be unreliable
                            st.markdown("**Kemungkinan Penyakit Lain:**")
                            top_n = min(3, len(label_encoder.classes_))
                            top_indices = np.argsort(pred_probs[0])[-top_n:][::-1]
                            
                            for i, idx in enumerate(top_indices):
                                if idx != pred_class_idx:
                                    disease_en = label_encoder.inverse_transform([idx])[0]
                                    disease_id = translate_to_indonesian(disease_en)
                                    prob = pred_probs[0][idx]
                                    st.write(f"- {disease_id} ({prob*100:.1f}%)")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
                finally:
                    # Clean up memory
                    if selected_model == "ANN":
                        tf.keras.backend.clear_session()
                    gc.collect()

# Data exploration section
expander = st.expander("Eksplorasi Data")
with expander:
    tab1, tab2 = st.tabs(["Contoh Data", "Distribusi Penyakit"])
    
    with tab1:
        st.subheader("Contoh Data Latih")
        st.write(data[['text', 'label']].head(10))
    
    with tab2:
        st.subheader("Distribusi Penyakit dalam Dataset")
        class_dist = data['label'].value_counts()
        
        # Translate class names for display
        dist_df = pd.DataFrame({
            'Penyakit': get_translated_diseases(class_dist.index),
            'Jumlah': class_dist.values
        })
        
        st.bar_chart(dist_df.set_index('Penyakit'))
        st.dataframe(dist_df, use_container_width=True)

# Add footer
st.markdown("---")
st.caption("""
Aplikasi Prediksi Penyakit © 2023 - Dibangun dengan Streamlit dan Scikit-learn
""")

# Always check cache at the end of script execution
manage_cache()