import streamlit as st
import numpy as np
import joblib

# ====== Konfigurasi Halaman ======
st.set_page_config(page_title="Prediksi Stres", layout="wide")

# ====== CSS Custom (Pink Lembut) ======
st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #fff0f5 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffe6f0 !important;
    }
    * {
        font-weight: bold !important;
        color: #804060 !important;
    }
    h1, h2, h3, h4, h5 {
        color: #804060 !important;
    }
    div[data-baseweb="input"] > div,
    .stNumberInput input {
        background-color: #fffafc !important;
        color: #804060 !important;
    }
    .stButton > button {
        background-color: #ff99cc !important;
        color: white !important;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #cc0066 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ====== Sidebar Navigasi ======
menu = st.sidebar.radio("📌 Navigasi", ["Home", "Prediksi", "Visualisasi"])

# ====== Halaman: HOME ======
if menu == "Home":
    st.title("👋 Selamat Datang di Aplikasi Prediksi Stres")
    st.write("""
        Aplikasi ini membantu memprediksi tingkat stres berdasarkan data seperti suhu, kelembaban, dan jumlah langkah harian.
        
        Silakan pilih menu **Prediksi** untuk mulai mengisi data atau lihat **Visualisasi** untuk melihat grafik dan informasi terkait stres.
    """)

# ====== Halaman: PREDIKSI ======
elif menu == "Prediksi":
    st.title("🧠 Prediksi Tingkat Stres")
    st.write("Masukkan data berikut untuk mengetahui tingkat stres kamu:")

    humidity = st.number_input("💧 Humidity(%)", min_value=0.0, max_value=100.0, step=0.1)
    temperature = st.number_input("🌡 Temperature (°C)", min_value=0.0, max_value=100.0, step=0.1)
    step_count = st.number_input("👣 Step count", min_value=0, step=1)

    if st.button("🔍 Prediksi"):
        try:
            model = joblib.load("Model_stres.pkl")
            scaler = joblib.load("scaler_stres.pkl")

            input_data = np.array([humidity, temperature, step_count]).reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            hasil = model.predict(input_scaled)[0]

            label_stres = {
                0: "😌 Tingkat stres kamu **Rendah**",
                1: "😵‍💫 Tingkat stres kamu **Sedang**",
                2: "🥵 Tingkat stres kamu **Tinggi**"
            }

            st.markdown(f"### Hasil Prediksi: {label_stres.get(hasil, '❓ Tidak dikenali')}")
        except FileNotFoundError:
            st.error("❌ File model atau scaler tidak ditemukan.")
        except Exception as e:
            st.error(f"⚠ Terjadi error saat memprediksi: {e}")

# ====== Halaman: VISUALISASI ======
elif menu == "Visualisasi":
    st.title("📊 Visualisasi Data Stres")
    st.write("Berikut adalah visualisasi confusion matrix dari model prediksi stres:")

    st.image("cnf.png", 
             caption="Confusion Matrix Model Prediksi Stres", 
             use_container_width=True)
