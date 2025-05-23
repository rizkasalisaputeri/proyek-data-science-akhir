import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Muat model yang sudah dilatih
model = joblib.load('model_dropout_prediction.pkl')

# Inisialisasi LabelEncoder untuk variabel kategorikal
le = LabelEncoder()

# Judul aplikasi
st.title("Prediksi Risiko Dropout Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi apakah mereka berisiko dropout (1) atau tidak (0).")

# Bagian Input: Demografi
st.subheader("Data Demografi")
gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
age_at_enrollment = st.slider("Usia Saat Mendaftar", 17, 70, 20)

# Bagian Input: Sosial-Ekonomi
st.subheader("Data Sosial-Ekonomi")
debtor = st.selectbox("Status Utang", ['Yes', 'No'])
tuition_fees_up_to_date = st.selectbox("Biaya Kuliah Terbayar", ['Yes', 'No'])
scholarship_holder = st.selectbox("Penerima Beasiswa", ['Yes', 'No'])

# Bagian Input: Akademik
st.subheader("Data Akademik")
admission_grade = st.slider("Nilai Penerimaan", 0, 200, 100)
curricular_units_1st_sem_approved = st.slider("Unit Disetujui Semester 1", 0, 20, 6)
curricular_units_1st_sem_enrolled = st.slider("Unit Terdaftar Semester 1", 0, 20, 6)
curricular_units_2nd_sem_approved = st.slider("Unit Disetujui Semester 2", 0, 20, 6)
curricular_units_2nd_sem_enrolled = st.slider("Unit Terdaftar Semester 2", 0, 20, 6)

# Buat data input dalam bentuk dictionary dengan nilai default untuk fitur yang tidak diinput
input_data = {
    'Marital_status': 'Single',  # Modus
    'Application_mode': '1st phase',  # Modus
    'Application_order': 1,  # Rata-rata
    'Course': 'Other',  # Modus
    'Daytime_evening_attendance': 'Daytime',  # Modus
    'Previous_qualification': 'High School',  # Modus
    'Previous_qualification_grade': 10,  # Rata-rata (sesuaikan dengan datasetmu)
    'Nacionality': 'Portuguese',  # Modus
    'Mothers_qualification': 'Basic',  # Modus
    'Fathers_qualification': 'Basic',  # Modus
    'Mothers_occupation': 'Other',  # Modus
    'Fathers_occupation': 'Other',  # Modus
    'Admission_grade': admission_grade,
    'Displaced': 'No',  # Modus
    'Educational_special_needs': 'No',  # Modus
    'Debtor': debtor,
    'Tuition_fees_up_to_date': tuition_fees_up_to_date,
    'Gender': gender,
    'Scholarship_holder': scholarship_holder,
    'Age_at_enrollment': age_at_enrollment,
    'International': 'No',  # Modus
    'Curricular_units_1st_sem_credited': 0,  # Rata-rata
    'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
    'Curricular_units_1st_sem_evaluations': 8,  # Rata-rata
    'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
    'Curricular_units_1st_sem_grade': 10,  # Rata-rata
    'Curricular_units_1st_sem_without_evaluations': 0,  # Rata-rata
    'Curricular_units_2nd_sem_credited': 0,  # Rata-rata
    'Curricular_units_2nd_sem_enrolled': curricular_units_2nd_sem_enrolled,
    'Curricular_units_2nd_sem_evaluations': 8,  # Rata-rata
    'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
    'Curricular_units_2nd_sem_grade': 10,  # Rata-rata
    'Curricular_units_2nd_sem_without_evaluations': 0,  # Rata-rata
    'Unemployment_rate': 10.0,  # Rata-rata
    'Inflation_rate': 1.0,  # Rata-rata
    'GDP': 0.0  # Rata-rata
}

# Konversi ke DataFrame
input_df = pd.DataFrame([input_data])

# Encode variabel kategorikal
categorical_cols = ['Marital_status', 'Application_mode', 'Course', 'Daytime_evening_attendance',
                   'Previous_qualification', 'Nacionality', 'Mothers_qualification', 'Fathers_qualification',
                   'Mothers_occupation', 'Fathers_occupation', 'Displaced', 'Educational_special_needs',
                   'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'International']
for col in categorical_cols:
    input_df[col] = le.fit_transform(input_df[col])

# Tambahkan fitur academic_risk (sesuai feature engineering di notebook)
input_df['academic_risk_1st_sem'] = input_df['Curricular_units_1st_sem_approved'] / input_df['Curricular_units_1st_sem_enrolled']
input_df['academic_risk_1st_sem'] = input_df['academic_risk_1st_sem'].fillna(0)
input_df['academic_risk_2nd_sem'] = input_df['Curricular_units_2nd_sem_approved'] / input_df['Curricular_units_2nd_sem_enrolled']
input_df['academic_risk_2nd_sem'] = input_df['academic_risk_2nd_sem'].fillna(0)

# Pastikan urutan kolom sesuai dengan yang digunakan saat melatih model
feature_names = ['Marital_status', 'Application_mode', 'Application_order', 'Course', 
                 'Daytime_evening_attendance', 'Previous_qualification', 'Previous_qualification_grade',
                 'Nacionality', 'Mothers_qualification', 'Fathers_qualification', 'Mothers_occupation', 
                 'Fathers_occupation', 'Admission_grade', 'Displaced', 'Educational_special_needs', 
                 'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 
                 'Age_at_enrollment', 'International', 'Curricular_units_1st_sem_credited', 
                 'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations', 
                 'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade', 
                 'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited', 
                 'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations', 
                 'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade', 
                 'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate', 
                 'Inflation_rate', 'GDP', 'academic_risk_1st_sem', 'academic_risk_2nd_sem']

# Debug: Periksa apakah semua fitur ada di input_df
missing_features = [col for col in feature_names if col not in input_df.columns]
if missing_features:
    st.error(f"Fitur yang hilang di input_df: {missing_features}")
else:
    # Pastikan urutan kolom sama
    input_df = input_df[feature_names]

    # Prediksi
    if st.button("Prediksi Risiko Dropout"):
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        st.subheader("Hasil Prediksi")
        if prediction[0] == 1:
            st.error("Mahasiswa ini **berisiko dropout** (Prediksi: 1)")
        else:
            st.success("Mahasiswa ini **tidak berisiko dropout** (Prediksi: 0)")
        
        st.subheader("Probabilitas")
        st.write(f"Probabilitas Dropout: {probability[0][1]:.2f}")
        st.write(f"Probabilitas Tidak Dropout: {probability[0][0]:.2f}")

# Tampilkan data input untuk verifikasi
st.subheader("Data Input yang Dimasukkan")
st.write(input_df)