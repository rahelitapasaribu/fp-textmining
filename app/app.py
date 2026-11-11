"""Sistem Klasifikasi Hate Speech Tweet Bahasa Indonesia (TF-IDF + SVM)"""

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = PROJECT_DIR / "model" / "best_svm_model.pkl"
VECTORIZER_PATH = PROJECT_DIR / "model" / "tfidf_vectorizer.joblib"

KAMUS_PATH = (
    PROJECT_DIR / "dataset" / "raw" / "new_kamusalay.csv"
)

METRICS_PATH = PROJECT_DIR / "model" / "model_info.json"

DB_PATH = BASE_DIR / "riwayat_klasifikasi.db"

LABEL_MAP = {
    0: "Non-Hate Speech",
    1: "Hate Speech"
}

st.set_page_config(
    page_title="Klasifikasi Hate Speech Pada Tweet Berbahasa Indonesia",
    page_icon="🛡️",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------

@st.cache_resource
def load_preprocessing_resources():

    # Load kamus normalisasi
    alay_dict = pd.read_csv(
        KAMUS_PATH,
        encoding="latin-1",
        header=None
    )

    alay_dict.columns = ["original", "replacement"]

    alay_dict_map = dict(
        zip(alay_dict["original"], alay_dict["replacement"])
    )

    # Stopword removal
    stopword_factory = StopWordRemoverFactory()
    stopwords = set(stopword_factory.get_stop_words())

    # Stemming
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()

    return alay_dict_map, stopwords, stemmer


def cleaning_text(text):

    text = str(text)

    def decode_utf8_escape(match):
        hex_values = re.findall(
            r'\\x([0-9a-fA-F]{2})',
            match.group()
        )

        byte_data = bytes(
            int(value, 16) for value in hex_values
        )

        return byte_data.decode(
            'utf-8',
            errors='replace'
        )

    text = re.sub(
        r'(?:\\x[0-9a-fA-F]{2})+',
        decode_utf8_escape,
        text
    )

    text = re.sub(r'\\[nrt]', ' ', text)
    text = re.sub(r'[\n\r\t]+', ' ', text)

    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' ',
        text
    )

    text = re.sub(
        r'#([A-Za-z0-9_]+)',
        r'\1',
        text
    )

    text = re.sub(
        r'\bRT\b',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r'[0-9]+', '', text)

    text = re.sub(
        r'[^\w\s]',
        ' ',
        text,
        flags=re.UNICODE
    )

    text = text.replace('_', ' ')

    text = re.sub(
        r'\bUSER\b',
        'USER',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r'(?:\bUSER\b\s*){2,}',
        'USER ',
        text
    )

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenizingText(text):
    # preserve_line=True menghindari kebutuhan download resource punkt
    return word_tokenize(text, preserve_line=True)


def normalizationText(tokens, alay_dict_map):

    normalized_tokens = []

    for word in tokens:
        replacement = alay_dict_map.get(word, word)
        normalized_tokens.extend(replacement.split())

    return normalized_tokens


def stopwordRemoval(tokens, stopwords):

    return [
        word for word in tokens
        if word not in stopwords
    ]


def stemmingText(tokens, stemmer):

    stems = [
        stemmer.stem(word)
        for word in tokens
    ]

    return ' '.join(stems)


def preprocess_text(text):

    alay_dict_map, stopwords, stemmer = (
        load_preprocessing_resources()
    )

    # 1. Cleaning
    cleaned_text = cleaning_text(text)

    # 2. Case folding
    casefolded_text = cleaned_text.lower()

    # 3. Tokenization
    tokens = tokenizingText(casefolded_text)

    # 4. Normalization
    normalized_tokens = normalizationText(
        tokens,
        alay_dict_map
    )

    # 5. Stopword removal
    filtered_tokens = stopwordRemoval(
        normalized_tokens,
        stopwords
    )

    # 6. Stemming
    stemmed_text = stemmingText(
        filtered_tokens,
        stemmer
    )

    return stemmed_text

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    # Jika model bukan Pipeline, load vectorizer terpisah
    if not hasattr(model, "named_steps"):
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        vectorizer = None

    return model, vectorizer


def _confidence(model, X):

    # Menggunakan probabilitas jika tersedia
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        return probabilities.max(axis=1)

    # Jika SVM tidak menggunakan probability=True,
    # gunakan skor keputusan sebagai perkiraan confidence.
    scores = model.decision_function(X)

    if scores.ndim == 1:
        return 1 / (1 + np.exp(-np.abs(scores)))

    exp_scores = np.exp(
        scores - scores.max(axis=1, keepdims=True)
    )

    probabilities = (
        exp_scores / exp_scores.sum(axis=1, keepdims=True)
    )

    return probabilities.max(axis=1)


def classify(texts: list[str]) -> pd.DataFrame:

    model, vectorizer = load_model()

    # Preprocessing seluruh teks
    processed_texts = [
        preprocess_text(text)
        for text in texts
    ]

    # Jika model adalah Pipeline TF-IDF + SVM,
    # masukkan teks langsung ke model.
    if vectorizer is None:
        X = processed_texts

    # Jika model hanya SVM, gunakan vectorizer terpisah.
    else:
        X = vectorizer.transform(processed_texts)

    # Prediksi
    predictions = model.predict(X)

    # Confidence score
    confidence = _confidence(model, X)

    labels = [
        LABEL_MAP.get(int(pred), str(pred))
        for pred in predictions
    ]

    return pd.DataFrame({
        "tweet": texts,
        "label": labels,
        "confidence": np.round(confidence, 4)
    })

# ----------------------------------------------------------------------------
# Database (SQLite)
# ----------------------------------------------------------------------------
def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS batch (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                waktu TEXT NOT NULL,
                jenis_input TEXT NOT NULL,      -- 'Teks' / 'CSV'
                nama_file TEXT,
                jumlah_data INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hasil (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL,
                tweet TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                FOREIGN KEY (batch_id) REFERENCES batch(id)
            );
            """
        )


def save_history(df: pd.DataFrame, jenis_input: str, nama_file: str | None = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO batch (waktu, jenis_input, nama_file, jumlah_data) VALUES (?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), jenis_input, nama_file, len(df)),
        )
        batch_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO hasil (batch_id, tweet, label, confidence) VALUES (?, ?, ?, ?)",
            [(batch_id, r.tweet, r.label, float(r.confidence)) for r in df.itertuples()],
        )
    return batch_id


def load_batches() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT id, waktu, jenis_input, nama_file, jumlah_data FROM batch ORDER BY id DESC",
            conn,
        )


def load_batch_detail(batch_id: int) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT tweet, label, confidence FROM hasil WHERE batch_id = ?",
            conn,
            params=(batch_id,),
        )


def delete_batch(batch_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM hasil WHERE batch_id = ?", (batch_id,))
        conn.execute("DELETE FROM batch WHERE id = ?", (batch_id,))


# ----------------------------------------------------------------------------
# Komponen UI
# ----------------------------------------------------------------------------
def show_single_result(row: pd.Series):
    is_hate = row["label"] == "Hate Speech"
    col1, col2 = st.columns(2)
    col1.metric("Hasil Klasifikasi", row["label"])
    col2.metric("Confidence Score", f"{row['confidence'] * 100:.2f}%")
    if is_hate:
        st.error("Tweet terdeteksi sebagai **Hate Speech**.")
    else:
        st.success("Tweet terdeteksi sebagai **Non-Hate Speech**.")
    st.progress(float(row["confidence"]))


def show_batch_result(df: pd.DataFrame):
    total = len(df)
    n_hate = int((df["label"] == "Hate Speech").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data", total)
    c2.metric("Hate Speech", n_hate)
    c3.metric("Non-Hate Speech", total - n_hate)

    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ----------------------------------------------------------------------------
# Halaman: Dashboard
# ----------------------------------------------------------------------------
def page_dashboard():
    st.title("🛡️ Dashboard Klasifikasi Hate Speech")
    st.caption("Klasifikasi tweet berbahasa Indonesia menggunakan TF-IDF dan Support Vector Machine (SVM).")

    tab_teks, tab_csv = st.tabs(["📝 Input Teks", "📁 Upload File CSV"])

    # ---- Input teks ----
    with tab_teks:
        tweet = st.text_area(
            "Masukkan tweet berbahasa Indonesia",
            height=150,
            placeholder="Ketik atau tempel tweet di sini...",
        )
        if st.button("Klasifikasikan", type="primary", key="btn_teks"):
            if not tweet.strip():
                st.warning("Teks tidak boleh kosong.")
            else:
                with st.spinner("Memproses..."):
                    result = classify([tweet.strip()])
                    save_history(result, jenis_input="Teks")
                st.session_state["hasil_teks"] = result.iloc[0]

        if "hasil_teks" in st.session_state:
            show_single_result(st.session_state["hasil_teks"])

    # ---- Input CSV ----
    with tab_csv:
        file = st.file_uploader("Unggah file CSV berisi kumpulan tweet", type=["csv"])
        if file is not None:
            try:
                df_in = pd.read_csv(file)
            except Exception as e:
                st.error(f"Gagal membaca file CSV: {e}")
                return

            st.write("Pratinjau data:")
            st.dataframe(df_in.head(), use_container_width=True)

            # Tebak kolom teks
            kandidat = [c for c in df_in.columns if c.lower() in ("tweet", "text", "teks", "content")]
            default_idx = list(df_in.columns).index(kandidat[0]) if kandidat else 0
            kolom = st.selectbox("Pilih kolom yang berisi tweet", df_in.columns, index=default_idx)

            if st.button("Klasifikasikan File", type="primary", key="btn_csv"):
                texts = df_in[kolom].dropna().astype(str).tolist()
                if not texts:
                    st.warning("Kolom yang dipilih tidak berisi data.")
                else:
                    with st.spinner(f"Memproses {len(texts)} tweet..."):
                        result = classify(texts)
                        save_history(result, jenis_input="CSV", nama_file=file.name)
                    st.session_state["hasil_csv"] = result
                    st.session_state["nama_csv"] = file.name

        if "hasil_csv" in st.session_state:
            st.subheader("Hasil Klasifikasi")
            hasil = st.session_state["hasil_csv"]
            show_batch_result(hasil)
            st.download_button(
                "⬇️ Unduh Hasil (CSV)",
                data=to_csv_bytes(hasil),
                file_name=f"hasil_klasifikasi_{st.session_state.get('nama_csv', 'data')}",
                mime="text/csv",
            )


# ----------------------------------------------------------------------------
# Halaman: Riwayat Klasifikasi
# ----------------------------------------------------------------------------
def page_riwayat():
    st.title("🕑 Riwayat Klasifikasi")

    batches = load_batches()
    if batches.empty:
        st.info("Belum ada riwayat klasifikasi.")
        return

    # Ringkasan
    st.subheader("Ringkasan Proses")
    st.dataframe(
        batches.rename(
            columns={
                "id": "ID",
                "waktu": "Waktu Input",
                "jenis_input": "Jenis Input",
                "nama_file": "Nama File",
                "jumlah_data": "Jumlah Data",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Detail
    st.subheader("Detail Hasil")
    pilihan = {
        f"#{r.id} | {r.waktu} | {r.jenis_input}"
        + (f" ({r.nama_file})" if r.nama_file else "")
        + f" | {r.jumlah_data} data": int(r.id)
        for r in batches.itertuples()
    }
    dipilih = st.selectbox("Pilih proses klasifikasi", list(pilihan.keys()))
    batch_id = pilihan[dipilih]

    detail = load_batch_detail(batch_id)
    show_batch_result(detail)

    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button(
        "⬇️ Unduh",
        data=to_csv_bytes(detail),
        file_name=f"riwayat_{batch_id}.csv",
        mime="text/csv",
    )
    if c2.button("🗑️ Hapus", key=f"hapus_{batch_id}"):
        delete_batch(batch_id)
        st.rerun()


# ----------------------------------------------------------------------------
# Halaman: Fitur Sistem
# ----------------------------------------------------------------------------
def page_fitur():
    st.title("⚙️ Fitur Sistem")

    st.markdown(
        """
        Sistem ini mengklasifikasikan tweet berbahasa Indonesia
        ke dalam dua kelas, yaitu **Hate Speech** dan
        **Non-Hate Speech**.

        **Alur proses klasifikasi:**
        1. Input data berupa teks langsung atau file CSV.
        2. Preprocessing teks.
        3. Ekstraksi fitur menggunakan TF-IDF.
        4. Klasifikasi menggunakan Support Vector Machine (SVM).
        5. Penyimpanan hasil ke database.
        """
    )

    st.divider()

    st.subheader("Hasil Evaluasi Model SVM")

    if not METRICS_PATH.exists():
        st.warning(
            "Hasil evaluasi belum tersedia. "
            "Silakan jalankan evaluasi model pada notebook "
            "dan simpan hasilnya ke model_info.json."
        )
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        m = json.load(f)

    # Menampilkan metrik evaluasi
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{m['test_accuracy'] * 100:.2f}%"
    )

    c2.metric(
        "Precision",
        f"{m['test_precision'] * 100:.2f}%"
    )

    c3.metric(
        "Recall",
        f"{m['test_recall'] * 100:.2f}%"
    )

    c4.metric(
        "F1-Score",
        f"{m['test_f1_score'] * 100:.2f}%"
    )

    st.divider()

    if "jumlah_data_uji" in m:
        st.caption(
            f"Evaluasi dilakukan menggunakan "
            f"{m['jumlah_data_uji']} data testing."
        )

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():

    init_db()

    if not MODEL_PATH.exists():
        st.error(
            f"File model tidak ditemukan: {MODEL_PATH}"
        )
        st.stop()

    try:
        model, vectorizer = load_model()

        if vectorizer is not None and not VECTORIZER_PATH.exists():
            st.error(
                f"File vectorizer tidak ditemukan: {VECTORIZER_PATH}"
            )
            st.stop()

    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    with st.sidebar:
        st.header("Hate Speech Classifier")

        halaman = st.radio(
            "Navigasi",
            ["Dashboard", "Riwayat Klasifikasi", "Fitur Sistem"],
            label_visibility="collapsed"
        )

        st.divider()
        st.caption("TF-IDF + SVM • Tweet Bahasa Indonesia")

    if halaman == "Dashboard":
        page_dashboard()

    elif halaman == "Riwayat Klasifikasi":
        page_riwayat()

    else:
        page_fitur()


if __name__ == "__main__":
    main()