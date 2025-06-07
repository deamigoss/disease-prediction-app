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
import sys

# Configuration - Adjusted for Streamlit Cloud limits
MEMORY_THRESHOLD_MB = 2000  # Warning threshold (2GB)
MEMORY_CRITICAL_MB = 2500   # Critical threshold (2.5GB)
CLEAR_CACHE_EVERY = 3       # Clear cache every N interactions
MAX_MODELS_IN_MEMORY = 1    # Only keep one model in memory
REBOOT_COOLDOWN = 60        # Minimum seconds between reboots
CHUNK_SIZE = 500            # Processing chunk size for memory efficiency

# Initialize session state for memory management
if 'interaction_count' not in st.session_state:
    st.session_state.interaction_count = 0
if 'last_cache_clear' not in st.session_state:
    st.session_state.last_cache_clear = time.time()
if 'last_reboot' not in st.session_state:
    st.session_state.last_reboot = 0

def get_memory_usage():
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / (1024 ** 2)

def soft_reboot():
    """Perform a soft reboot of the Streamlit app"""
    current_time = time.time()
    
    # Check if we're in a cooldown period
    if current_time - st.session_state.last_reboot < REBOOT_COOLDOWN:
        # Show cooldown message without blocking
        cooldown = st.empty()
        remaining = int(REBOOT_COOLDOWN - (current_time - st.session_state.last_reboot))
        cooldown.warning(f"Reboot on cooldown. Please wait {remaining} seconds...")
        time.sleep(1)
        cooldown.empty()
        return False
    
    # Mark the reboot time
    st.session_state.last_reboot = current_time
    
    # Clear all caches and states
    st.cache_data.clear()
    st.cache_resource.clear()
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Set reboot flag and clear critical variables
    st.session_state.reboot_requested = True
    
    # IMPORTANT: Clear only necessary session state, keep essential config
    keys_to_keep = ['interaction_count', 'last_cache_clear', 'last_reboot']
    new_state = {k: st.session_state.get(k) for k in keys_to_keep}
    st.session_state.clear()
    st.session_state.update(new_state)
    
    # Rerun without delay to prevent UI stuck
    st.rerun()

def manage_memory():
    """Monitor and manage memory usage"""
    mem = get_memory_usage()
    st.session_state.interaction_count += 1
    
    # Check if we need to reboot
    if mem > MEMORY_CRITICAL_MB:
        soft_reboot()
        return
    
    # Check if we should clear cache
    if (st.session_state.interaction_count % CLEAR_CACHE_EVERY == 0) or (mem > MEMORY_THRESHOLD_MB):
        # Clear various caches
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Clear TensorFlow session if it exists
        if 'current_model' in st.session_state and st.session_state.get('current_model_type') == "ANN":
            tf.keras.backend.clear_session()
        
        # Force garbage collection
        gc.collect()
        
        # Update tracking variables
        st.session_state.last_cache_clear = time.time()
        
        # Show notification
        reason = "memory threshold" if mem > MEMORY_THRESHOLD_MB else "periodic schedule"
        st.toast(f"🧹 Cache cleared ({reason}) - Memory: {mem:.1f}MB", icon="✅")

@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')

# --- Text Preprocessing ---
def preprocess_text(text):
    """Lightweight text preprocessing"""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    
    # Only stem if text is short to save memory
    if len(text.split()) < 50:
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
        return ' '.join(tokens)
    return text

# --- Load & Preprocessing ---
@st.cache_data(show_spinner="Memuat dataset...")
def load_data():
    # Load only necessary columns in chunks
    cols = ['text', 'label']
    chunks = pd.read_csv('dataset.csv', usecols=cols, chunksize=CHUNK_SIZE)
    
    processed_chunks = []
    for chunk in chunks:
        chunk = chunk.dropna(subset=['text', 'label'])
        chunk['processed_text'] = chunk['text'].apply(preprocess_text)
        processed_chunks.append(chunk)
        gc.collect()
    
    return pd.concat(processed_chunks)

# --- Main App ---
def main():
    try:
        # Initial memory check before anything else
        if get_memory_usage() > MEMORY_CRITICAL_MB * 0.8:
            soft_reboot()
            return
        
        # Handle reboot sequence
        if st.session_state.get('reboot_requested', False):
            # Clear the flag first
            del st.session_state.reboot_requested
            
            # Show reboot message in a container
            reboot_container = st.container()
            with reboot_container:
                st.success("✅ App successfully rebooted! Loading fresh session...")
                
                # Use progress bar instead of sleep
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                
                reboot_container.empty()
            
            st.rerun()
            return

        # Load NLTK resources
        download_nltk_resources()
        
        # Load data
        data = load_data()
        
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
        tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # --- Model Selection ---
        model_options = {
            "ANN": "Neural Network dengan Embedding",
            "SVM": "Support Vector Machine",
            "Random Forest": "Random Forest",
            "Naive Bayes": "Multinomial Naive Bayes"
        }

        # --- Setup Translator ---
        translator = Translator()

        @st.cache_data(ttl=3600, max_entries=1000)
        def translate_to_english(text):
            try:
                time.sleep(0.5)  # Prevent rate limiting
                result = translator.translate(text, src='id', dest='en')
                return result.text
            except Exception as e:
                st.error(f"Error translating to English: {e}")
                return text

        @st.cache_data(ttl=3600, max_entries=1000)
        def translate_to_indonesian(text):
            try:
                time.sleep(0.5)
                result = translator.translate(text, src='en', dest='id')
                return result.text
            except Exception as e:
                st.error(f"Error translating to Indonesian: {e}")
                return text

        # --- Streamlit UI ---
        st.title("🩺 Sistem Prediksi Penyakit Berbasis Gejala")
        st.markdown("""
        Aplikasi ini menggunakan teknik NLP untuk memprediksi penyakit berdasarkan deskripsi gejala yang Anda berikan.
        """)

        # Sidebar for model selection and info
        with st.sidebar:
            st.header("Pengaturan Model")
            
            selected_model = st.selectbox(
                "Pilih Model Klasifikasi",
                list(model_options.keys()),
                index=0
            )

            st.markdown("---")
            st.markdown("**Informasi Dataset:**")
            st.write(f"Jumlah Data: {len(data)}")
            st.write(f"Jumlah Penyakit: {len(label_encoder.classes_)}")

            st.markdown("---")
            st.markdown("**Daftar Penyakit:**")

            # Translate disease labels to Indonesian
            translated_classes = []
            for disease in label_encoder.classes_:  # Limit to first 20 for performance
                try:
                    translated = translate_to_indonesian(disease)
                    translated_classes.append(translated)
                except:
                    translated_classes.append(disease)
            
            # Show diseases in sidebar
            df_labels_id = pd.DataFrame({'Penyakit': translated_classes})
            st.dataframe(df_labels_id, use_container_width=True)

        # --- Model Training ---
        if 'model_metrics' not in st.session_state:
            st.session_state.model_metrics = {}

        if selected_model != st.session_state.get('current_model_type'):
            # Clear previous model
            with st.spinner('Membersihkan model sebelumnya...'):
                if 'current_model' in st.session_state:
                    if st.session_state.current_model_type == "ANN":
                        tf.keras.backend.clear_session()
                    del st.session_state.current_model
                    del st.session_state.current_model_type
                gc.collect()
                manage_memory()

            # Train new model with memory constraints
            try:
                with st.spinner(f'Melatih model {selected_model}...'):
                    start_time = time.time()
                    
                    # Memory check before training
                    if get_memory_usage() > MEMORY_THRESHOLD_MB:
                        soft_reboot()
                        return
                    
                    # Train in chunks for ANN
                    if selected_model == "ANN":
                        model = Sequential([
                            Dense(64, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
                            Dropout(0.3),
                            Dense(32, activation='relu'),
                            Dense(len(label_encoder.classes_), activation='softmax')
                        ])
                        
                        model.compile(
                            optimizer=Adam(learning_rate=0.001),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        
                        # Train in smaller batches
                        batch_size = 32
                        for i in range(0, X_train_tfidf.shape[0], CHUNK_SIZE):
                            chunk = X_train_tfidf[i:i+CHUNK_SIZE].toarray()
                            y_chunk = y_train[i:i+CHUNK_SIZE]
                            model.fit(
                                chunk, y_chunk,
                                epochs=5,
                                batch_size=batch_size,
                                verbose=0
                            )
                            gc.collect()
                    else:
                        # For other models
                        model = {
                            "SVM": SVC(kernel='linear', probability=True, cache_size=200),
                            "Random Forest": RandomForestClassifier(n_estimators=30, max_depth=5),
                            "Naive Bayes": MultinomialNB()
                        }[selected_model]
                        model.fit(X_train_tfidf, y_train)
                    
                    # Store model
                    st.session_state.current_model = model
                    st.session_state.current_model_type = selected_model
                    
                    # Evaluate model
                    with st.spinner('Evaluasi model...'):
                        if selected_model == "ANN":
                            y_pred = model.predict(X_test_tfidf.toarray()).argmax(axis=1)
                        else:
                            y_pred = model.predict(X_test_tfidf)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        waktu_sekarang = datetime.now(pytz.timezone('Asia/Jakarta'))
                        format_waktu = waktu_sekarang.strftime("%Y-%m-%d %H:%M:%S")
                        
                        st.session_state.model_metrics[selected_model] = {
                            'accuracy': accuracy,
                            'training_time': time.time() - start_time,
                            'last_trained': format_waktu
                        }
                    
                    st.success(f"Model {selected_model} berhasil dilatih!")
            
            except Exception as e:
                st.error(f"Gagal melatih model: {str(e)}")
                if 'current_model' in st.session_state:
                    del st.session_state.current_model
                if 'current_model_type' in st.session_state:
                    del st.session_state.current_model_type
                gc.collect()

        # Get the current model
        current_model = st.session_state.get('current_model')

        # Display model info if available
        if current_model is not None and selected_model in st.session_state.model_metrics:
            metrics = st.session_state.model_metrics[selected_model]
            
            st.subheader("Informasi Model")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model", selected_model)
            
            with col2:
                st.metric("Akurasi", f"{metrics['accuracy']*100:.1f}%")
            
            with col3:
                st.metric("Waktu Training", f"{metrics['training_time']:.1f} detik")
            
            st.caption(f"Terakhir dilatih: {metrics['last_trained']}")

        # User input section
        st.subheader("Masukkan Gejala Anda")
        user_input = st.text_area(
            "Deskripsikan gejala yang Anda alami:",
            placeholder="Contoh: Saya mengalami demam tinggi, sakit kepala, dan nyeri otot..."
        )

        if st.button("Prediksi Penyakit"):
            manage_memory()
            if not user_input:
                st.warning("Silakan masukkan deskripsi gejala terlebih dahulu.")
            elif current_model is None:
                st.error("Model belum dilatih. Silakan tunggu hingga proses training selesai.")
            else:
                with st.spinner('Memproses gejala dan membuat prediksi...'):
                    try:
                        # Memory check before prediction
                        if get_memory_usage() > MEMORY_THRESHOLD_MB:
                            soft_reboot()
                            return
                        
                        # Translate
                        input_en = translate_to_english(user_input)
                        
                        # Preprocess
                        processed_input = preprocess_text(input_en)
                        input_tfidf = tfidf_vectorizer.transform([processed_input])
                        
                        # Predict
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
                        st.markdown(f"**Penyakit:** {pred_class_id}")
                        st.markdown(f"**Tingkat Kepercayaan:** {confidence*100:.1f}%")
                        
                        # Show top predictions if available
                        if selected_model != "SVM" and len(pred_probs[0]) > 1:
                            top_n = min(3, len(label_encoder.classes_))
                            top_indices = np.argsort(pred_probs[0])[-top_n:][::-1]
                            
                            for i, idx in enumerate(top_indices[1:], 1):  # Skip first (main prediction)
                                disease_en = label_encoder.inverse_transform([idx])[0]
                                disease_id = translate_to_indonesian(disease_en)
                                prob = pred_probs[0][idx]
                                st.write(f"{i}. {disease_id} ({prob*100:.1f}%)")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
                    finally:
                        if selected_model == "ANN":
                            tf.keras.backend.clear_session()
                        gc.collect()

        # --- Data Exploration Section ---
        expander = st.expander("Eksplorasi Data")
        with expander:
            tab1, tab2 = st.tabs(["Contoh Data", "Distribusi Penyakit"])
            
            with tab1:
                st.subheader("Contoh Data Latih")
                st.write(data[['text', 'label']].sample(5, random_state=42))  # Consistent sample
            
            with tab2:
                st.subheader("Distribusi Penyakit dalam Dataset")
                
                # Calculate class distribution (top 10 only)
                class_dist = data['label'].value_counts().nlargest(10).reset_index()
                class_dist.columns = ['Penyakit', 'Jumlah']
                
                # Show visualization and table
                st.bar_chart(class_dist.set_index('Penyakit'))
                st.dataframe(class_dist, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan kritis: {str(e)}")
        if get_memory_usage() > MEMORY_THRESHOLD_MB:
            soft_reboot()

if __name__ == "__main__":
    main()