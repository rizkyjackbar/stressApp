import streamlit as st
import numpy as np
import pandas as pd
import pickle
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

st.set_page_config(page_title="StressPredict", page_icon=":chart_with_upwards_trend:")

def save_to_google_sheets(user_name, stress_level_label):
    try:
        print("Memulai proses menyimpan data ke Google Sheets...")

        # Scope untuk Google Sheets API
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

        # Load kredensial dari secrets Streamlit
        creds_dict = dict(st.secrets["google_credentials"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        print("Kredensial berhasil dimuat.")

        # Autentikasi dan akses client
        client = gspread.authorize(creds)
        print("Google Sheets API berhasil diotorisasi.")

        # Akses spreadsheet
        sheet = client.open("StressPredictResults").sheet1
        print("Spreadsheet berhasil diakses.")

        # Tambahkan data ke spreadsheet
        sheet.append_row([user_name, stress_level_label])
        print(f"Data berhasil disimpan: {user_name}, {stress_level_label}")
    except Exception as e:
        print(f"Terjadi kesalahan saat menyimpan ke Google Sheets: {e}")

# Load model dan scaler
with open('stress_level_model.pkl', 'rb') as f:
    saved_objects = pickle.load(f)
model = saved_objects['model']
scaler = saved_objects['scaler']

# Load dataset
url = 'StressLevelDataset.csv'
data = pd.read_csv(url)
X = data.drop(['stress_level', 'anxiety_level'], axis=1)

# Streamlit UI
st.title("Prediksi Tingkat Stres")
st.write("Jawab pertanyaan berikut untuk memprediksi tingkat stres kamu.")

user_name = st.text_input("Boleh dong, ketik nama kamu dulu!", placeholder="Contoh: Jackbar")

# User input nama
if not user_name.strip():
    st.warning("Tak kenal maka tak sayang lohhh, hehehe")
else:
    # Input user
    user_input = []
    min_max_values = {
        "self_esteem": (0, 100),
        "mental_health_history": (0, 1),
        "depression": (0, 100),
        "headache": (0, 100),
        "blood_pressure": (0, 100),
        "sleep_quality": (0, 100),
        "breathing_problem": (0, 100),
        "noise_level": (0, 100),
        "living_conditions": (0, 100),
        "safety": (0, 100),
        "basic_needs": (0, 100),
        "academic_performance": (0, 100),
        "study_load": (0, 100),
        "teacher_student_relationship": (0, 100),
        "future_career_concerns": (0, 100),
        "social_support": (0, 100),
        "peer_pressure": (0, 100),
        "extracurricular_activities": (0, 100),
        "bullying": (0, 100)
    }

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

    category_descriptions = {
        "self_esteem": ["Nggak percaya diri", "Cukup percaya diri", "Sangat percaya diri"],
        "depression": ["Jarang merasa depresi", "Kadang-kadang merasa depresi", "Sering merasa depresi"],
        "headache": ["Jarang sakit kepala", "Kadang-kadang sakit kepala", "Sering sakit kepala"],
        "blood_pressure": ["Rendah", "Sedang", "Tinggi"],
        "sleep_quality": ["Buruk", "Sedang", "Baik"],
        "breathing_problem": ["Jarang", "Kadang-kadang", "Sering"],
        "noise_level": ["Tenang", "Sedang bising", "Sangat bising"],
        "living_conditions": ["Nggak layak", "Cukup layak", "Sangat layak"],
        "safety": ["Nggak aman", "Cukup aman", "Sangat aman"],
        "basic_needs": ["Nggak terpenuhi", "Sebagian terpenuhi", "Terpenuhi"],
        "academic_performance": ["Buruk", "Sedang", "Baik"],
        "study_load": ["Ringan", "Sedang", "Berat"],
        "teacher_student_relationship": ["Buruk", "Cukup baik", "Sangat baik"],
        "future_career_concerns": ["Tidak khawatir", "Cukup khawatir", "Sangat khawatir"],
        "social_support": ["Tidak ada dukungan", "Sedikit dukungan", "Banyak dukungan"],
        "peer_pressure": ["Tidak pernah", "Kadang-kadang", "Sering"],
        "extracurricular_activities": ["Tidak aktif", "Cukup aktif", "Sangat aktif"],
        "bullying": ["Tidak pernah", "Kadang-kadang", "Sering"]
    }
    
    stress_level_advice = {
        "Ringan": "Stres kamu masih ringan kok! Tetap jaga pola hidup sehat ya, seperti makan teratur, tidur cukup, dan jangan lupa gerak badan. Santai aja, semua pasti baik-baik aja!",
        "Sedang": "Hmm, stres kamu ada di tingkat sedang nih. Coba deh cari waktu buat istirahat, ngobrol sama teman atau keluarga, atau coba relaksasi kayak meditasi atau denger musik. Jangan dipendem sendiri, ya!",
        "Berat": "Wah, stres kamu udah di tingkat berat nih. Jangan anggap enteng ya. Kalau bisa, segera ngobrol sama psikolog atau konselor. Kadang kita perlu bantuan orang lain buat lepas dari tekanan. Kamu nggak sendiri kok!"
    }

    stress_level_emoji = {
        "Ringan": "😊",
        "Sedang": "😐",
        "Berat": "😟"
    }

    for col in X.columns:
        if col == "mental_health_history":
            value = st.radio(
                f"{questions[col]}",
                options=[0, 1],
                format_func=lambda x: "Nggak punya" if x == 0 else "Punya"
            )
        else:
            min_val, max_val = min_max_values[col]
            value = st.slider(f"{questions[col]}", min_val, max_val, step=1, format="%d%%")

            # Label kategori
            num_categories = len(category_descriptions[col])
            category_idx = min((value - min_val) * num_categories // (max_val - min_val + 1), num_categories - 1)
            st.write(f"Keterangan: {category_descriptions[col][category_idx]}")

        user_input.append(value)
        st.divider()
        
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
