import streamlit as st
import numpy as np
import pandas as pd
import pickle
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="StressPredict", page_icon=":chart_with_upwards_trend:")

def save_to_google_sheets(user_name, stress_level_label):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credentials_path = "credentials.json"
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
        client = gspread.authorize(creds)
        sheet = client.open("StressPredictResults").sheet1
        sheet.append_row([user_name, stress_level_label])
        print(f"Data berhasil disimpan ke Google Sheets: {user_name}, {stress_level_label}")
    except Exception as e:
        print(f"Error saving to Google Sheets: {e}")

# Load model dan scaler dari .pkl
with open('stress_level_model.pkl', 'rb') as f:
    saved_objects = pickle.load(f)
model = saved_objects['model']
scaler = saved_objects['scaler']

# Load dataset
url = 'StressLevelDataset.csv'
data = pd.read_csv(url)
X = data.drop(['stress_level', 'anxiety_level'], axis=1)

# Pertanyaan dan kategori deskripsi
questions = {
    "self_esteem": "Seberapa percaya diri kamu sama kemampuan dirimu?",
    "mental_health_history": "Kamu punya riwayat gangguan kesehatan mental nggak?",
    "depression": "Seberapa sering kamu merasa depresi belakangan ini?",
    "headache": "Seberapa sering kamu ngerasa sakit kepala?",
    "blood_pressure": "Gimana tekanan darah kamu akhir-akhir ini?",
    "sleep_quality": "Gimana kualitas tidur kamu?",
    "breathing_problem": "Kamu pernah ngalamin masalah pernapasan nggak?",
    "noise_level": "Seberapa sering kamu terganggu sama kebisingan di lingkungan kamu?",
    "living_conditions": "Gimana kondisi tempat tinggal kamu?",
    "safety": "Seberapa aman kamu ngerasa di lingkungan sekitar?",
    "basic_needs": "Kebutuhan dasar kamu terpenuhi nggak?",
    "academic_performance": "Gimana performa akademik kamu sekarang?",
    "study_load": "Seberapa banyak beban belajar kamu?",
    "teacher_student_relationship": "Gimana hubungan kamu sama guru atau dosen?",
    "future_career_concerns": "Seberapa khawatir kamu sama masa depan karier kamu?",
    "social_support": "Seberapa besar dukungan sosial yang kamu dapet dari orang-orang sekitar?",
    "peer_pressure": "Seberapa sering kamu merasa tertekan sama teman sebaya?",
    "extracurricular_activities": "Seberapa aktif kamu di kegiatan ekstrakurikuler?",
    "bullying": "Kamu pernah ngalamin bullying nggak?"
}

category_descriptions = {
    "self_esteem": ["Nggak percaya diri", "Cukup percaya diri", "Sangat percaya diri"],
    "depression": ["Jarang merasa depresi", "Kadang-kadang merasa depresi", "Sering merasa depresi"],
    "headache": ["Jarang sakit kepala", "Kadang-kadang sakit kepala", "Sering sakit kepala"],
    "blood_pressure": ["Normal", "Tinggi", "Sangat tinggi"],
    "sleep_quality": ["Buruk", "Sedang", "Baik"],
    "breathing_problem": ["Jarang", "Kadang-kadang", "Sering"],
    "noise_level": ["Tenang", "Sedang bising", "Sangat bising"],
    "living_conditions": ["Nggak layak", "Cukup layak", "Sangat layak"],
    "safety": ["Nggak aman", "Cukup aman", "Sangat aman"],
    "basic_needs": ["Nggak terpenuhi", "Sebagian terpenuhi", "Terpenuhi"],
    "academic_performance": ["Buruk", "Sedang", "Baik"],
    "study_load": ["Ringan", "Sedang", "Berat"],
    "teacher_student_relationship": ["Buruk", "Cukup baik", "Sangat baik"],
    "future_career_concerns": ["Nggak khawatir", "Cukup khawatir", "Sangat khawatir"],
    "social_support": ["Nggak ada dukungan", "Sedikit dukungan", "Banyak dukungan"],
    "peer_pressure": ["Nggak pernah", "Kadang-kadang", "Sering"],
    "extracurricular_activities": ["Nggak aktif", "Cukup aktif", "Sangat aktif"],
    "bullying": ["Nggak pernah", "Kadang-kadang", "Sering"]
}

# Streamlit UI
st.title("Prediksi Tingkat Stres")
st.write("Jawab pertanyaan berikut untuk memprediksi tingkat stres kamu.")

# Input nama pengguna
user_name = st.text_input("Masukkan nama kamu", placeholder="Contoh: Jackbar")

if not user_name.strip():
    st.warning("Yukk tulis nama kamu dulu.")
else:
    user_input = []
    min_max_values = {col: (data[col].min(), data[col].max()) for col in X.columns}

    for col in X.columns:
        if col in category_descriptions:
            value = st.slider(f"{questions[col]}", 0, len(category_descriptions[col]) - 1, step=1)
            st.write(f"Keterangan: {category_descriptions[col][value]}")
        else:
            min_val, max_val = int(min_max_values[col][0]), int(min_max_values[col][1])
            value = st.slider(f"{questions[col]} (min: {min_val}, max: {max_val})", min_val, max_val, step=1)
        user_input.append(value)

        # Tambahkan pembatas antar pertanyaan
        st.divider()

    if st.button("Prediksi"):
        user_input = np.array(user_input).reshape(1, -1)
        user_input = scaler.transform(user_input)
        prediction = model.predict(user_input)
        stress_level = np.argmax(prediction)

        stress_level_map = {0: 'Ringan', 1: 'Sedang', 2: 'Berat'}
        stress_level_label = stress_level_map[stress_level]

        st.success(f"Hasil Prediksi: Tingkat stres Anda adalah **{stress_level_label}**.")
