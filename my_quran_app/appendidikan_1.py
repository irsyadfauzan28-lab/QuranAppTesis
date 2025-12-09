import streamlit as st
import pandas as pd
import nltk
import string
import difflib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download resource NLTK
nltk.download('punkt')
nltk.download('stopwords')

# ----------------- PREPROCESSING ----------------- #
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('indonesian')]
    return ' '.join(tokens)

def correct_word(word, vocab):
    matches = difflib.get_close_matches(word, vocab, n=1, cutoff=0.7)
    return matches[0] if matches else word

def spell_correct_query(query, vocab):
    query = query.lower()
    query = query.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(query)
    corrected = [correct_word(word, vocab) for word in tokens if word not in stopwords.words('indonesian')]
    return ' '.join(corrected)

# ----------------- LOAD DATA ----------------- #
@st.cache_data
def load_quran():
    df = pd.read_excel("quran.xlsx")
    df['processed_text'] = df['translation'].apply(preprocess_text)
    return df

@st.cache_data
def load_book():
    book_df = pd.read_excel("Rangkuman_Bab_Berdasarkan_Ayat_dan_Hadits.xlsx")
    book_df['processed_text'] = book_df['Isi Pokok'].fillna("").apply(preprocess_text)
    return book_df

# ----------------- SEARCH FUNCTION ----------------- #
def search(query, tfidf_matrix, vectorizer, df, vocab, top_n=5):
    corrected_query = spell_correct_query(query, vocab)
    query_vector = vectorizer.transform([corrected_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# ----------------- STREAMLIT UI ----------------- #
st.set_page_config(page_title="Aplikasi Quran & Pembelajaran", layout="centered")
st.title("📘 Aplikasi Pencarian Ayat Quran & Materi Pendidikan Islam")

# Load data
df_quran = load_quran()
df_book = load_book()

# Buat vocab & vectorizer
vocab = set(" ".join(df_quran['processed_text']).split() + " ".join(df_book['processed_text']).split())
vectorizer = TfidfVectorizer()
tfidf_quran = vectorizer.fit_transform(df_quran['processed_text'])
tfidf_book = vectorizer.transform(df_book['processed_text'])

# Input pengguna
query = st.text_input("Masukkan kata kunci (boleh typo):")

if query:
    st.subheader("📖 Hasil Pencarian dari Quran")
    results_quran = search(query, tfidf_quran, vectorizer, df_quran, vocab)
    for _, row in results_quran.iterrows():
        st.markdown(f"**{row['surat']} : {row['ayat']}**")
        st.markdown(f"{row['translation']}")
        st.markdown("---")

    st.subheader("📚 Hasil Pencarian dari Buku Pendidikan Islam")
    results_book = search(query, tfidf_book, vectorizer, df_book, vocab)
    for _, row in results_book.iterrows():
        st.markdown(f"**Bab {row['Bab']} | Judul: {row['Judul']}**")  # Perbaikan kolom "Judul"
        st.markdown(f"{row['Isi Pokok']}")
        st.markdown("---")
