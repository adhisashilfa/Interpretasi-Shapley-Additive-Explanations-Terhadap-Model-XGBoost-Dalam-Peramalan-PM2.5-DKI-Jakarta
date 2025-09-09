🌫️ Prediksi Kualitas Udara Menggunakan XGBoost
Proyek ini membangun model machine learning untuk memprediksi kualitas udara (PM2.5) menggunakan algoritma XGBoost. Model ini di-deploy melalui aplikasi Flask untuk memberikan prediksi dan insight berbasis data.

🚀 Latar Belakang
Kualitas udara berperan penting dalam kesehatan masyarakat dan lingkungan. Pemantauan manual sering kali terlambat dalam memberikan informasi. Oleh karena itu, diperlukan sistem prediksi berbasis machine learning agar masyarakat dan pembuat kebijakan dapat mengambil keputusan lebih cepat.

🎯 Tujuan
- Memprediksi konsentrasi PM2.5 berdasarkan data sensor polutan (CO, O3, SO2, NO2, dll).
- Memberikan visualisasi tren kualitas udara harian.
- Menginterpretasikan hasil model menggunakan SHAP values untuk memahami faktor dominan.

🛠️ Teknologi & Tools
- Bahasa: Python
- Framework / Library: Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn
- Deployment: Flask
- Tools: Jupyter Notebook, Git, VSCode

🔍 Langkah Pengerjaan
- Data Collection: Menggunakan data sensor kualitas udara dari [sumber dataset].
- Data Preprocessing: Handling missing values, outlier treatment (winsorizing), scaling, encoding.
- EDA: Tren harian PM2.5 & polutan lain, heatmap korelasi, distribusi data.
- Modeling: Hyperparameter tuning dengan GridSearchCV pada XGBoost Regressor.
- Evaluation: Menggunakan R², RMSE, MAPE pada data uji (20% & 30%).
- Interpretasi: SHAP summary plot, dependence plot, interaction values.
- Deployment: Membuat aplikasi Flask sederhana untuk prediksi real-time dan visualisasi.

📊 Hasil & Insight
- Model XGBoost mencapai R² = 0.87 pada data uji.
- Faktor paling berpengaruh terhadap PM2.5: CO, NO2, dan O3.
- Visualisasi menunjukkan tren peningkatan PM2.5 pada musim kemarau.
- Aplikasi Flask memudahkan pengguna dalam melakukan prediksi dengan input parameter.

🌟 Impact / Achievements
- Memberikan sistem prediksi yang dapat membantu pemantauan kualitas udara.
- Meningkatkan efisiensi analisis dibandingkan metode manual.
- Menunjukkan potensi penerapan AI untuk lingkungan di Indonesia.

🤝 Kontribusi & Peran
- Data preprocessing dan EDA
- Pengembangan model XGBoost dengan hyperparameter tuning
- Analisis interpretasi model menggunakan SHAP
- Pembuatan aplikasi Flask + dashboard visualisasi
