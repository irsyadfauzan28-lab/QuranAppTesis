import streamlit as st
import pandas as pd
import nltk
import string
import difflib
import fitz # PyMuPDF
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download resource NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing teks
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('indonesian')]
    return ' '.join(tokens)

# Load data Quran
@st.cache_data
def load_data():
    df = pd.read_excel("quran.xlsx")
    df['processed_text'] = df['translation'].apply(preprocess_text)
    return df

# Load data buku pendidikan
@st.cache_data
def load_buku_pendidikan(path="Islam_BS_KLS_XII_.pdf"):
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        text = page.get_text()
        full_text += text + " "
    cleaned_text = preprocess_text(full_text)
    return cleaned_text

# Koreksi ejaan
def correct_word(word, vocab):
    matches = difflib.get_close_matches(word, vocab, n=1, cutoff=0.7)
    return matches[0] if matches else word

def spell_correct_query(query, vocab):
    query = query.lower()
    query = query.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(query)
    corrected = [correct_word(word, vocab) for word in tokens if word not in stopwords.words('indonesian')]
    return ' '.join(corrected)

# Fungsi pencarian gabungan
def search_combined(query, combined_matrix, vectorizer, df, buku_text, vocab, top_n=5):
    corrected_query = spell_correct_query(query, vocab)
    query_vector = vectorizer.transform([corrected_query])
    similarities = cosine_similarity(query_vector, combined_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        if idx < len(df):
            sumber = "Quran"
            data = df.iloc[idx]
            results.append((sumber, f"{data['surat']} : {data['ayat']}", data['translation']))
        else:
            sumber = "Buku Pendidikan"
            results.append((sumber, "-", buku_text[:300] + "..."))
    return results

# Streamlit UI
st.set_page_config(page_title="Quran Search App", layout="centered")
st.title("🔍 Aplikasi Pencarian Ayat Quran dan Literasi Pendidikan")

# Load data
df = load_data()
buku_text = load_buku_pendidikan()

# Gabungkan korpus dan vektorisasi
vocab = set(" ".join(df['processed_text']).split())
combined_corpus = df['processed_text'].tolist() + [buku_text]
combined_vectorizer = TfidfVectorizer()
combined_tfidf_matrix = combined_vectorizer.fit_transform(combined_corpus)

# Input pengguna
query = st.text_input("Masukkan kata kunci (boleh typo):")

if query:
    st.write("Hasil Pencarian:")
    results = search_combined(query, combined_tfidf_matrix, combined_vectorizer, df, buku_text, vocab)

    for sumber, ayat, teks in results:
        st.markdown(f"**{sumber} | {ayat}**")
        st.markdown(f"{teks}")
        st.markdown("---")
