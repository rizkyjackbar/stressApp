import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

st.set_page_config(page_title="StressPredict", page_icon=":chart_with_upwards_trend:")

def save_to_google_sheets(user_name, stress_level_label):
    try:
        # Scope untuk Google Sheets API
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

        # Path ke file credentials.json
        credentials_path = "credentials.json"

        # Load kredensial dari file JSON
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
        client = gspread.authorize(creds)

        # Akses spreadsheet
        sheet = client.open("StressPredictResults").sheet1

        # Tambahkan data ke spreadsheet
        sheet.append_row([user_name, stress_level_label])
        print(f"Data berhasil disimpan ke Google Sheets: {user_name}, {stress_level_label}")
    except Exception as e:
        print(f"Error saving to Google Sheets: {e}")

# Load model dan scaler dari .pkl
with open('stress_level_model.pkl', 'rb') as f:
    saved_objects = pickle.load(f)
model = saved_objects['model']
scaler = saved_objects['scaler']

# Load dataset untuk informasi kolom dan skala
url = 'StressLevelDataset.csv'
data = pd.read_csv(url)
X = data.drop(['stress_level', 'anxiety_level'], axis=1)

# Streamlit UI
st.title("Prediksi Tingkat Stres")
st.write("Jawab pertanyaan berikut untuk memprediksi tingkat stres kamu.")

# Input nama pengguna
user_name = st.text_input("Boleh dong, ketik nama kamu dulu!", placeholder="Contoh: Jackbar")

# Memastikan user menginputkan nama
if not user_name.strip():
    st.warning("Tak kenal maka tak sayang lohhh, hehehe")
else:
    # Input dari pengguna
    user_input = []
    min_max_values = {col: (data[col].min(), data[col].max()) for col in X.columns}

    # Pertanyaan spesifik untuk setiap faktor
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
        "basic_needs": "Kebutuhan dasar kamu (makanan, pakaian, tempat tinggal) terpenuhi nggak?",
        "academic_performance": "Gimana performa akademik kamu sekarang?",
        "study_load": "Seberapa banyak beban belajar kamu?",
        "teacher_student_relationship": "Gimana hubungan kamu sama guru atau dosen?",
        "future_career_concerns": "Seberapa khawatir kamu sama masa depan karier kamu?",
        "social_support": "Seberapa besar dukungan sosial yang kamu dapet dari orang-orang sekitar?",
        "peer_pressure": "Seberapa sering kamu merasa tertekan sama teman sebaya?",
        "extracurricular_activities": "Seberapa aktif kamu di kegiatan ekstrakurikuler?",
        "bullying": "Kamu pernah ngalamin bullying nggak?"
    }

    stress_level_advice = {
        "Ringan": "Stres kamu masih ringan kok! Tetap jaga pola hidup sehat ya, seperti makan teratur, tidur cukup, dan jangan lupa gerak badan. Santai aja, semua pasti baik-baik aja!",
        "Sedang": "Hmm, stres kamu ada di tingkat sedang nih. Coba deh cari waktu buat istirahat, ngobrol sama teman atau keluarga, atau coba relaksasi kayak yoga atau meditasi. Jangan dipendem sendiri, ya!",
        "Berat": "Wah, stres kamu udah di tingkat berat nih. Jangan anggap enteng ya. Kalau bisa, segera ngobrol sama psikolog atau konselor. Kadang kita perlu bantuan orang lain buat lepas dari tekanan. Kamu nggak sendiri kok!"
    }

    stress_level_emoji = {
        "Ringan": "😊",
        "Sedang": "😐",
        "Berat": "😟"
    }

    for col in X.columns:
        if col == "mental_health_history":
            value = st.radio(f"{questions[col]}", options=[0, 1], format_func=lambda x: "Nggak punya" if x == 0 else "Punya")
        else:
            min_val, max_val = int(min_max_values[col][0]), int(min_max_values[col][1])
            value = st.slider(f"{questions[col]}", min_val, max_val, step=1)

        user_input.append(value)

    if st.button("Prediksi"):
        user_input = np.array(user_input).reshape(1, -1)
        user_input = scaler.transform(user_input)

        prediction = model.predict(user_input)
        stress_level = prediction[0]

        stress_level_map = {0: 'Ringan', 1: 'Sedang', 2: 'Berat'}
        stress_level_label = stress_level_map[stress_level]

        save_to_google_sheets(user_name, stress_level_label)

        color_map = {"Ringan": "green", "Sedang": "yellow", "Berat": "red"}
        color = color_map[stress_level_label]

        st.markdown(
            f"""
            <div style="border: 2px solid {color}; padding: 20px; border-radius: 10px; text-align: center; background-color: {color}33;">
                <div style="font-size: 72px;">{stress_level_emoji[stress_level_label]}</div>
                <h2 style="color: {color};">Haiii kak {user_name}!! tingkat stres nya: {stress_level_label}</h2>
                <p>{stress_level_advice[stress_level_label]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
