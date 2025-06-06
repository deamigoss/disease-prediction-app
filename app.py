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


@st.cache_resource
def download_nltk_resources():
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
@st.cache_data(max_entries=1)
def load_data():
    dtype = {
        'column1': 'category',
        'column2': 'float32'
    }
    return pd.read_csv('data.csv', dtype=dtype, usecols=['col1', 'col2'])

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

def translate_to_english(text):
    try:
        time.sleep(0.5)  # Prevent rate limiting
        result = translator.translate(text, src='id', dest='en')
        return result.text
    except Exception as e:
        st.error(f"Error translating to English: {e}")
        return text

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


# Fungsi monitor memori
def memory_usage_widget():
    col1, col2, col3 = st.columns(3)
    
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    sys_mem = psutil.virtual_memory()
    
    col1.metric("App Memory Used", f"{mem.rss / (1024 ** 2):.1f} MB")
    col2.metric("System Available", f"{sys_mem.available / (1024 ** 3):.1f} GB")
    col3.metric("Memory Usage (%)", f"{sys_mem.percent}%")

# Panggil di bagian utama app
memory_usage_widget()



# Initialize session state for models
if 'models' not in st.session_state:
    st.session_state.models = {}
    st.session_state.current_model = None

# Train selected model only if not already trained
if selected_model not in st.session_state.models:
    if selected_model == "ANN":
        with st.spinner('Melatih model ANN...'):
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
    
    elif selected_model == "SVM":
        with st.spinner('Melatih model SVM...'):
            model = SVC(kernel='linear', probability=True)
            model.fit(X_train_tfidf, y_train)
            st.session_state.models["SVM"] = model
            st.session_state.current_model = "SVM"
    
    elif selected_model == "Random Forest":
        with st.spinner('Melatih model Random Forest...'):
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train_tfidf, y_train)
            st.session_state.models["Random Forest"] = model
            st.session_state.current_model = "Random Forest"
    
    elif selected_model == "Naive Bayes":
        with st.spinner('Melatih model Naive Bayes...'):
            model = MultinomialNB()
            model.fit(X_train_tfidf, y_train)
            st.session_state.models["Naive Bayes"] = model
            st.session_state.current_model = "Naive Bayes"

# Get the current model
current_model = st.session_state.models.get(selected_model)

# Display model performance only if we have a trained model
if current_model is not None:
    st.subheader("Evaluasi Model")
    
    if selected_model == "ANN":
        y_pred_probs = current_model.predict(X_test_tfidf.toarray())
        y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        y_pred = current_model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Akurasi pada Data Test: **{accuracy:.2f}**")

# User input section
st.subheader("Masukkan Gejala Anda")
user_input = st.text_area(
    "Deskripsikan gejala yang Anda alami dalam bahasa Indonesia:",
    placeholder="Contoh: Saya mengalami demam tinggi, sakit kepala, dan nyeri otot..."
)

if st.button("Prediksi Penyakit"):
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

# Show sample data
if st.checkbox("Tampilkan Contoh Data"):
    st.subheader("Contoh Data Latih")
    st.write(data[['text', 'label']].head(10))

# Show class distribution
if st.checkbox("Tampilkan Distribusi Kelas"):
    st.subheader("Distribusi Penyakit dalam Dataset")
    class_dist = data['label'].value_counts()
    st.bar_chart(class_dist)