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
import time


MEMORY_THRESHOLD_MB = 500  # Set your memory threshold (MB)
CLEAR_CACHE_EVERY = 5     # Clear cache every N interactions


# Initialize session state for cache management
if 'interaction_count' not in st.session_state:
    st.session_state.interaction_count = 0
if 'last_cache_clear' not in st.session_state:
    st.session_state.last_cache_clear = time.time()

def manage_cache():
    """Intelligently manage cache based on interactions and memory usage"""
    st.session_state.interaction_count += 1
    
    # Check memory usage using psutil
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

@st.cache_resource
def download_nltk_resources(ttl = 3600):
    nltk.download('stopwords')

download_nltk_resources()


# --- Text Preprocessing ---
def preprocess_text(text):
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


# --- Load & Preprocessing ---
@st.cache_data(max_entries=1, ttl=3600, show_spinner="Memuat dataset...")
def load_data():
    # Load only necessary columns
    cols = ['text', 'label']
    data = pd.read_csv('dataset.csv', usecols=cols)
    
    # Drop NA values
    data = data.dropna(subset=['text', 'label'])
    
    # Preprocess in chunks if dataset is large
    chunk_size = 1000
    processed_texts = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        processed_chunk = chunk['text'].apply(preprocess_text)
        processed_texts.extend(processed_chunk)
    
    data['processed_text'] = processed_texts
    return data

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
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
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

# --- Streamlit App ---
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


# --- Model Building Functions ---
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
    st.session_state.current_model_type = None 
    st.session_state.model_metrics = {}

def build_ann_model(input_dim, output_dim):
    """Build and compile ANN model with optimized architecture"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),  # Reduced from 256
        Dropout(0.3),  # Reduced from 0.5
        Dense(64, activation='relu'),  # Reduced from 128
        Dropout(0.2),  # Reduced from 0.3
        Dense(output_dim, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model_type, X_train, y_train, num_classes):
    """Generic model training function with memory optimization"""
    if model_type == "ANN":
        model = build_ann_model(X_train.shape[1], num_classes)
        early_stopping = EarlyStopping(patience=2, restore_best_weights=True)  # Reduced patience
        
        # Train with smaller batch size
        history = model.fit(
            X_train.toarray(), y_train,
            epochs=50,  
            batch_size=32, 
            validation_split=0.1,  
            callbacks=[early_stopping],
            verbose=0
        )
        return model
    
    elif model_type == "SVM":
        model = SVC(kernel='linear', probability=True, cache_size=200)  # Limited cache size
        model.fit(X_train, y_train)
        return model
    
    elif model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,   
            n_jobs=1         
        )
        model.fit(X_train, y_train)
        return model
    
    elif model_type == "Naive Bayes":
        model = MultinomialNB()
        model.fit(X_train, y_train)
        return model

# --- Model Management in Session State ---
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
    st.session_state.current_model_type = None
    st.session_state.model_metrics = {}

# Check if model needs to be changed or trained
if selected_model != st.session_state.current_model_type:
    # Clear previous model
    with st.spinner('Membersihkan model sebelumnya...'):
        if st.session_state.current_model_type == "ANN":
            tf.keras.backend.clear_session()
        st.session_state.current_model = None
        gc.collect()
        manage_cache()  # Clear cache before loading new model
    
    # Train new model
    try:
        with st.spinner(f'Melatih model {selected_model}...'):
            start_time = time.time()
            
            # Get the appropriate training data
            if selected_model == "ANN":
                train_data = X_train_tfidf.toarray()
            else:
                train_data = X_train_tfidf
            
            # Train the model
            model = train_model(
                selected_model,
                train_data,
                y_train,
                len(label_encoder.classes_)
            )
            
            # Store in session state
            st.session_state.current_model = model
            st.session_state.current_model_type = selected_model
            
            # Calculate and store training time
            training_time = time.time() - start_time
            st.session_state.model_metrics[selected_model] = {
                'training_time': training_time,
                'last_trained': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Evaluate model
            with st.spinner('Evaluasi model...'):
                if selected_model == "ANN":
                    y_pred_probs = model.predict(X_test_tfidf.toarray())
                    y_pred = np.argmax(y_pred_probs, axis=1)
                else:
                    y_pred = model.predict(X_test_tfidf)
                
                accuracy = accuracy_score(y_test, y_pred)
                st.session_state.model_metrics[selected_model]['accuracy'] = accuracy
            
            st.success(f"Model {selected_model} berhasil dilatih!")
            manage_cache()  # Clear cache after training
    
    except Exception as e:
        st.error(f"Gagal melatih model: {str(e)}")
        st.session_state.current_model = None
        st.session_state.current_model_type = None

# Get the current model
current_model = st.session_state.current_model

# Display model info if available
if current_model is not None and selected_model in st.session_state.model_metrics:
    metrics = st.session_state.model_metrics[selected_model]
    
    st.subheader("Informasi Model")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model", selected_model)
    
    with col2:
        st.metric("Akurasi", f"{metrics.get('accuracy', 0)*100:.1f}%")
    
    with col3:
        st.metric("Waktu Training", f"{metrics.get('training_time', 0):.1f} detik")
    
    st.caption(f"Terakhir dilatih: {metrics.get('last_trained', 'N/A')}")


# User input section
st.subheader("Masukkan Gejala Anda")
user_input = st.text_area(
    "Deskripsikan gejala yang Anda alami dalam bahasa Indonesia:",
    placeholder="Contoh: Saya mengalami demam tinggi, sakit kepala, dan nyeri otot..."
)

if st.button("Prediksi Penyakit"):
    manage_cache()
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
                st.markdown(f"**Penyakit:** {pred_class_id}")
                st.markdown(f"**Tingkat Kepercayaan:** {confidence*100:.1f}%")
                
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
                # Bersihkan memory
                if selected_model == "ANN":
                    tf.keras.backend.clear_session()
                gc.collect()
if st.session_state.interaction_count % 3 == 0:
    manage_cache()

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