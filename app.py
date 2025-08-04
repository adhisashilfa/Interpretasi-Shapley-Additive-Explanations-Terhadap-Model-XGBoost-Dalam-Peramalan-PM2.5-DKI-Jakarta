import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shap
import io, base64
from flask import Flask, request, render_template, redirect, url_for, session, flash
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer


# Pastikan Matplotlib tidak menggunakan backend GUI
matplotlib.use("Agg")

app = Flask(__name__,static_folder="assets")
app.secret_key = "9dfaa9239f12239a3521dc666fd384f7"
UPLOAD_FOLDER = "data"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Fungsi preprocessing
def preprocess_data(df):
    df_clean = df.copy()

    # Simpan kolom tanggal jika ada
    tanggal = df_clean.pop("tanggal") if "tanggal" in df_clean.columns else None

    # Kolom kategorikal
    categorical_mapped = {
        "critical": {"PM10": 1, "PM2.5": 2, "SO2": 3, "CO": 4, "O3": 5, "NO2": 6}
    }

    # Critical: modus + mapping manual
    if "critical" in df_clean.columns:
        df_clean["critical"] = df_clean["critical"].fillna(df_clean["critical"].mode()[0])
        df_clean["critical"] = df_clean["critical"].replace(categorical_mapped["critical"])

    # Categori & stasiun: modus + label encoding
    label_enc_cols = ["categori", "stasiun"]
    for col in label_enc_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    # Tangani kolom numerik: replace 0 dengan NaN
    # Pastikan hanya kolom numerik non-kategorikal yang diproses
    kategori_kolom = ["critical", "categori", "stasiun"]
    numeric_cols = [col for col in df_clean.select_dtypes(include=[np.number]).columns if col not in kategori_kolom]

    df_clean[numeric_cols] = df_clean[numeric_cols].replace(0, np.nan)

    # Imputasi KNN hanya untuk kolom numerik
    imputer = KNNImputer(n_neighbors=5)
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])

    # Kembalikan kolom tanggal
    if tanggal is not None:
        df_clean.insert(0, "tanggal", tanggal)

    return df_clean

@app.route("/", methods=["GET", "POST"])
def welcome():
    return render_template("welcome.html")

@app.route("/upload_file", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            flash("⚠️ Tidak ada file yang diunggah!", "danger")
            return redirect(url_for("upload_file"))

        file = request.files["file"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        session["filepath"] = filepath

        return redirect(url_for("show_data"))
    
    return render_template("upload.html")

@app.route("/show_data")
def show_data():
    filepath = session.get("filepath")
    if not filepath:
        return redirect(url_for("upload_file"))
    
    df = pd.read_csv(filepath, sep=None, engine="python")
    table_html = df.head(10).to_html(classes="table table-bordered", index=False)
    
    return render_template("show_data.html", table_html=table_html)

@app.route("/preprocess", methods=["GET"])
def preprocess():
    filepath = session.get("filepath", None)
    if not filepath:
        return redirect(url_for("upload_file"))

    # Load the CSV file
    df = pd.read_csv(filepath, sep=None, engine="python")

    # Apply your cleaning function
    df["tanggal"] = pd.to_datetime(df["tanggal"], dayfirst=True, errors="coerce")

    df_clean = preprocess_data(df)

    # Save the cleaned data
    preprocessed_path = os.path.join(app.config["UPLOAD_FOLDER"], "preprocessed_data.csv")
    df_clean.to_csv(preprocessed_path, index=False)

    session["preprocessed_filepath"] = preprocessed_path

    # Render a preview
    table_html = df_clean.head(10).to_html(classes="table table-bordered", index=False)
    return render_template("preprocess.html", table_html=table_html)

@app.route("/eda", methods=["GET"])
def eda():
    import io, base64
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    from flask import render_template, session

    preprocessed_path = session.get("preprocessed_filepath", None)
    if not preprocessed_path or not os.path.exists(preprocessed_path):
        return "Error: Data preprocessing belum dilakukan atau file tidak ditemukan", 400

    df_clean = pd.read_csv(preprocessed_path)

    # Statistik dasar
    stats_html = df_clean.describe().to_html(classes="table table-bordered", index=True)

    # Missing values
    missing_values = df_clean.isnull().sum()
    missing_html = missing_values[missing_values > 0].to_frame().to_html(classes="table table-bordered")

    # Visualisasi tren harian
    series_images = {}
    pollutants = ["pm10", "pm2.5", "co", "o3", "so2", "no2"]

    df_clean["tanggal"] = pd.to_datetime(df_clean["tanggal"], errors="coerce")
    df_sorted = df_clean.sort_values("tanggal", ascending=True)
    palette = list(sns.palettes.mpl_palette("Dark2"))

    for pol in pollutants:
        if pol not in df_sorted.columns:
            continue

        img_buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(10, 5.2))
        for i, (name, group) in enumerate(df_sorted.groupby("stasiun")):
            ax.plot(group["tanggal"], group[pol], label=name, color=palette[i % len(palette)])

        ax.set_title(f"Tren Harian {pol.upper()} per Stasiun")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel(pol.upper())
        fig.legend(title="Stasiun", bbox_to_anchor=(1, 1), loc="upper left")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(img_buf, format="png", bbox_inches="tight")
        img_buf.seek(0)
        series_images[pol] = base64.b64encode(img_buf.getvalue()).decode()
        plt.close(fig)

    return render_template(
        "eda.html",
        stats_html=stats_html,
        missing_html=missing_html,
        series_pm10_encoded=series_images.get("pm10"),
        series_pm25_encoded=series_images.get("pm2.5"),
        series_co_encoded=series_images.get("co"),
        series_o3_encoded=series_images.get("o3"),
        series_so2_encoded=series_images.get("so2"),
        series_no2_encoded=series_images.get("no2")
    )

@app.route("/select_features", methods=["GET", "POST"])
def select_features():
    preprocessed_path = session.get("preprocessed_filepath")
    if not preprocessed_path:
        flash("⚠️ Data preprocessing belum dilakukan!", "danger")
        return redirect(url_for("preprocess"))
    
    df_clean = pd.read_csv(preprocessed_path)
    columns = df_clean.columns.tolist()
    
    if request.method == "POST":
        target_variable = request.form.get("target_variable")
        selected_features = request.form.getlist("features")

        if not target_variable or not selected_features:
            flash("⚠️ Harap pilih fitur dan target!", "danger")
            return redirect(url_for("select_features"))

        session["target_variable"] = target_variable
        session["selected_features"] = selected_features
        return redirect(url_for("train_config"))
    
    return render_template("select_features.html", columns=columns)

@app.route("/train_config", methods=["GET", "POST"])
def train_config():
    if request.method == "POST":
        test_size = float(request.form.get("test_size", 0.2))
        learning_rate = request.form.get("learning_rate")
        max_depth = request.form.get("max_depth")
        n_estimators = request.form.get("n_estimators")

        session["test_size"] = test_size
        session["learning_rate"] = [float(x) for x in learning_rate.split(',') if x]
        session["max_depth"] = [int(x) for x in max_depth.split(',') if x]
        session["n_estimators"] = [int(x) for x in n_estimators.split(',') if x]

        return redirect(url_for("predict"))
    
    return render_template("train_config.html")

@app.route("/predict", methods=["GET"])
def predict():
    preprocessed_path = session.get("preprocessed_filepath")
    if not preprocessed_path or not os.path.exists(preprocessed_path):
        flash("⚠️ Data preprocessing belum dilakukan!", "danger")
        return redirect(url_for("preprocess"))
    
    df_clean = pd.read_csv(preprocessed_path)
    selected_features = session.get("selected_features")
    target_variable = session.get("target_variable")
    test_size = session.get("test_size", 0.2)

    X = df_clean[selected_features]
    y = df_clean[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    param_grid = {
        "n_estimators": session.get("n_estimators"),
        "max_depth": session.get("max_depth"),
        "learning_rate": session.get("learning_rate")
    }

    model = XGBRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=10)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    rmse_value = np.sqrt(mean_squared_error(y_test, y_pred))

    comparison_df = pd.DataFrame({
        "Tanggal": pd.to_datetime(df_clean.loc[y_test.index, "tanggal"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "Actual": y_test.values,
        "Predicted": y_pred
    })
    predictions_html = comparison_df.head(10).to_html(classes="table table-bordered", index=False)

    plot_img = io.BytesIO()
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df['Actual'].reset_index(drop=True), label='Actual', color='blue')
    plt.plot(comparison_df['Predicted'].reset_index(drop=True), label='Predicted', color='red')
    plt.title('Actual vs Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_img, format='png', bbox_inches='tight')
    plot_img.seek(0)
    pred_plot_encoded = base64.b64encode(plot_img.getvalue()).decode()

    # Future prediction
    # Prediksi 3 Hari ke Depan
    df_clean['tanggal'] = pd.to_datetime(df_clean['tanggal'])

    # Buat fitur lag
    df_lag = df_clean[['tanggal', 'pm2.5']].copy()
    df_lag['lag_1'] = df_lag['pm2.5'].shift(1)
    df_lag['lag_2'] = df_lag['pm2.5'].shift(2)
    df_lag['lag_3'] = df_lag['pm2.5'].shift(3)
    df_lag.dropna(inplace=True)

    X_lag = df_lag[['lag_1', 'lag_2', 'lag_3']]
    y_lag = df_lag['pm2.5']
    X_train_lag, X_test_lag, y_train_lag, y_test_lag = train_test_split(X_lag, y_lag, test_size=test_size, shuffle=False)

    model_lag = XGBRegressor()
    model_lag.fit(X_train_lag, y_train_lag)

    # Prediksi 3 hari ke depan
    last_values = list(df_lag['pm2.5'].values[-3:])
    future_predictions = []
    for i in range(3):
        input_lag = np.array(last_values[-3:]).reshape(1, -1)
        next_pred = model_lag.predict(input_lag)[0]
        future_predictions.append(round(next_pred, 2))
        last_values.append(next_pred)

    last_date = df_lag['tanggal'].iloc[-1]
    future_dates_lag = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]
    future_lag_df = pd.DataFrame({
        'Tanggal': future_dates_lag,
        'Prediksi_PM25': future_predictions
    })

    # Plot hasil prediksi
    plt.figure(figsize=(10,5))
    plt.plot(df_lag['tanggal'].tail(7), df_lag['pm2.5'].tail(7), marker='o', label='Aktual PM2.5')
    plt.plot(future_lag_df['Tanggal'], future_lag_df['Prediksi_PM25'], marker='o', linestyle='--', label='Prediksi PM2.5')
    plt.title('Prediksi PM2.5 Tiga Hari ke Depan')
    plt.xlabel('Tanggal')
    plt.ylabel('Konsentrasi PM2.5 (µg/m³)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Simpan plot ke buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode()
    buf.close()

    # Inisialisasi SHAP explainer dan hitung SHAP values
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    shap_bar_img = io.BytesIO()
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(shap_bar_img, format="png", bbox_inches="tight")
    shap_bar_encoded = base64.b64encode(shap_bar_img.getvalue()).decode()

    shap_img = io.BytesIO()
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(shap_img, format="png", bbox_inches="tight")
    shap_encoded = base64.b64encode(shap_img.getvalue()).decode()

    return render_template("predict.html",
        rmse_value=f"{rmse_value:.2f}%",
        pred_plot_encoded=pred_plot_encoded,
        predictions=predictions_html,
        shap_bar_encoded=shap_bar_encoded,
        shap_encoded=shap_encoded,
        tables=future_lag_df.to_html(classes='table table-striped', index=False),
        plot_url=plot_data)


if __name__ == "__main__":
    app.run(debug=True)
