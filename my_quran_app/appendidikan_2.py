import streamlit as st
import pandas as pd
import nltk
import string
import difflib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForSequenceClassification
from torch import nn
import torch

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

# ----------------- BERT SEARCH ----------------- #
# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Function to compute cosine similarity between query and corpus
def bert_search(query, corpus):
    inputs = tokenizer([query] * len(corpus), corpus, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarities = nn.functional.cosine_similarity(outputs.logits[0], outputs.logits[1])
    return similarities

# Function to search relevant ayats using BERT
def search_ayat(query, corpus, ayat_list):
    similarities = bert_search(query, corpus)
    best_match_idx = torch.argmax(similarities).item()
    best_match_ayat = ayat_list[best_match_idx]
    return best_match_ayat

# ----------------- LOAD DATA ----------------- #
@st.cache_data
def load_quran():
    df = pd.read_excel("https://raw.githubusercontent.com/irsyadfauzan28-lab/QuranAppTesis/main/my_quran_app/quran.xlsx")
    df['processed_text'] = df['translation'].apply(preprocess_text)
    return df

@st.cache_data
def load_book():
    book_df = pd.read_excel("https://raw.githubusercontent.com/irsyadfauzan28-lab/QuranAppTesis/main/my_quran_app/Rangkuman_Bab_Berdasarkan_Ayat_dan_Hadits.xlsx")
    book_df['processed_text'] = book_df['Isi Pokok'].fillna("").apply(preprocess_text)
    return book_df

# ----------------- SEARCH FUNCTION ----------------- #
def search(query, tfidf_matrix, vectorizer, df, vocab, top_n=5):
    corrected_query = spell_correct_query(query, vocab)
    query_vector = vectorizer.transform([corrected_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# ----------------- GET KEYWORD EXPLANATION ----------------- #
def get_keyword_explanation(keyword, book_df):
    keyword = keyword.lower()
    explanations = book_df[book_df['processed_text'].str.contains(keyword)]
    if explanations.empty:
        return f"Tidak ada penjelasan mengenai kata kunci '{keyword}' dalam buku."
    else:
        return explanations[['Bab', 'Judul', 'Isi Pokok']]

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
query = st.text_input("Masukkan kata kunci:")

if query:
    st.subheader("Hasil Pencarian dari Quran")
    results_quran = search(query, tfidf_quran, vectorizer, df_quran, vocab)
    for _, row in results_quran.iterrows():
        st.markdown(f"**{row['surat']} : {row['ayat']}**")
        st.markdown(f"{row['translation']}")
        st.markdown("---")

    st.subheader("Hasil Pencarian dari Buku Pendidikan Islam")
    results_book = search(query, tfidf_book, vectorizer, df_book, vocab)
    for _, row in results_book.iterrows():
        st.markdown(f"**Bab {row['Bab']} | Judul: {row['Judul']}**")  
        st.markdown(f"{row['Isi Pokok']}")
        st.markdown("---")

    # ----------------- Keyword Explanation ----------------- #
    st.subheader("Penjelasan mengenai kata kunci")
    explanation = get_keyword_explanation(query, df_book)
    if isinstance(explanation, str):
        st.write(explanation)  # No explanation found
    else:
        for _, row in explanation.iterrows():
            st.write(f"**Bab**: {row['Bab']}")
            st.write(f"**Judul**: {row['Judul']}")
            st.write(f"**Isi Pokok**: {row['Isi Pokok']}")
            st.markdown("---")

# ----------------- BERT PENCARIAN ----------------- #
if st.button('Cari Ayat dengan BERT'):
    st.write(f"Pencarian dengan menggunakan model BERT untuk kata kunci: {query}")
    ayat_found = search_ayat(query, [ayat['translation'] for ayat in df_quran.to_dict('records')], df_quran['translation'].tolist())
    st.write(f"Ayat yang ditemukan: {ayat_found}")

# ----------------- FEEDBACK ----------------- #
st.write("Apakah pencarian ayat ini relevan dengan topik yang Anda cari?")
feedback = st.radio('Pilih salah satu', ['Relevan', 'Tidak Relevan'])

if feedback:
    st.write(f"Terima kasih atas feedback Anda! Anda memilih: {feedback}")
