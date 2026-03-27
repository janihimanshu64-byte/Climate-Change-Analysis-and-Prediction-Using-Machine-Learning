# Weather Forecast System - Web App
# Connects: base.html, login.html, upload.html, results.html, sample_dataset_10years.csv

import os
import io
import base64
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'weather-forecast-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
USERS_FILE = os.path.join(BASE_DIR, 'users.json')
SAMPLE_10YEARS = os.path.join(BASE_DIR, 'sample_dataset_10years.csv')
# Legacy source used only to auto-generate the 10-year sample (if needed).
SAMPLE_100YEARS = os.path.join(BASE_DIR, 'sample_dataset_100years.csv')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ADMIN_USER = os.environ.get('ADMIN_USER', 'admin')
ADMIN_PASS = os.environ.get('ADMIN_PASS', 'admin123')


def load_users():
    if os.path.isfile(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


def normalize_column_names(df):
    mapping = {}
    seen = set()
    for col in df.columns:
        c = str(col).strip().lower()
        target = None
        if 'date' in c or 'day' in c:
            target = 'date'
        elif 'temp' in c or 'temperature' in c:
            target = 'temperature'
        elif 'humid' in c:
            target = 'humidity'
        elif 'rain' in c or 'precip' in c:
            target = 'rainfall'
        elif 'wind' in c or 'speed' in c:
            target = 'wind_speed'
        elif 'condition' in c or 'weather' in c:
            target = 'condition'
        if target and target not in seen:
            mapping[col] = target
            seen.add(target)
    return df.rename(columns=mapping)


def ensure_sample_10years():
    """
    Create `sample_dataset_10years.csv` from the legacy 100-year sample (if needed).
    The app UI/code is intended to use only the last 10 years.
    """
    if os.path.isfile(SAMPLE_10YEARS):
        return True
    if not os.path.isfile(SAMPLE_100YEARS):
        return False

    try:
        df = pd.read_csv(SAMPLE_100YEARS)
        df = normalize_column_names(df)

        if 'date' in df.columns:
            d = pd.to_datetime(df['date'], errors='coerce')
            years = d.dt.year.dropna().astype(int).unique().tolist()
            if years:
                last_years = sorted(set(years))[-10:]
                mask = d.dt.year.isin(last_years)
                df = df.loc[mask].copy()
                df['date'] = d.loc[mask]

        df.to_csv(SAMPLE_10YEARS, index=False)
        return True
    except Exception:
        return False


def parse_uploaded_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 3:
            return None, "Dataset must have at least 3 rows."
        df = normalize_column_names(df)
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception:
                pass
        return df, None
    except Exception as e:
        return None, str(e)


def get_series(df, name, default=None):
    n = len(df)
    if n == 0:
        return np.array([])
    if name in df.columns:
        col = df[name]
        if hasattr(col, 'iloc') and getattr(col, 'ndim', 1) > 1:
            col = col.iloc[:, 0]
        s = pd.to_numeric(col, errors='coerce')
        s = s.ffill().bfill().fillna(0)
        out = np.asarray(s.values, dtype=float).ravel()
        return out[:n] if len(out) >= n else np.pad(out, (0, n - len(out)), mode='edge')
    if default is not None:
        arr = np.asarray(default, dtype=float).ravel()
        return arr[:n] if len(arr) >= n else np.pad(arr, (0, n - len(arr)), mode='edge')
    return np.zeros(n)


def get_available_years(df):
    if df is None or 'date' not in df.columns:
        return []
    try:
        d = pd.to_datetime(df['date'], errors='coerce')
        years = d.dt.year.dropna().astype(int).unique().tolist()
        return sorted(set(years))
    except Exception:
        return []


def filter_df_by_year(df, year):
    if year is None or year == '' or str(year).lower() == 'all':
        return df
    if df is None or 'date' not in df.columns:
        return df
    try:
        d = pd.to_datetime(df['date'], errors='coerce')
        mask = d.dt.year == int(year)
        return df.loc[mask].reset_index(drop=True)
    except Exception:
        return df


def filter_df_to_last_n_years(df, n=10):
    years = get_available_years(df)
    if not years or len(years) <= n:
        return df, years

    last_years = years[-n:]
    if df is None or 'date' not in df.columns:
        return df, last_years

    try:
        d = pd.to_datetime(df['date'], errors='coerce')
        mask = d.dt.year.isin(last_years)
        return df.loc[mask].reset_index(drop=True), last_years
    except Exception:
        return df, last_years


GRAPH_OPTIONS = [
    ('historical_temp', 'Historical Temperature Trend '),
    ('actual_vs_predicted', 'Actual vs Predicted Temperature'),
    ('humidity', 'Humidity Trend '),
    ('rainfall', 'Rainfall Analysis '),
    ('wind_speed', 'Wind Speed Variation'),
    ('distribution', 'Weather Condition Distribution'),
    ('error_analysis', 'Error Analysis'),
]


def generate_all_graphs(df, predicted_temp=None, which_graphs=None):
    if df is None or len(df) < 1:
        raise ValueError("Dataset has no rows.")
    want_all = not which_graphs or len(which_graphs) == 0

    def include(key):
        return want_all or (key in which_graphs)

    n_df = len(df)
    dates = df['date'].values if 'date' in df.columns else np.arange(n_df)
    if hasattr(dates, 'shape') and len(dates.shape) > 1:
        dates = dates.ravel()[:n_df]
    temperature = get_series(df, 'temperature', 25 + 5 * np.sin(np.linspace(0, 3, n_df)) + np.random.normal(0, 1, n_df))
    humidity = get_series(df, 'humidity', 60 + 10 * np.sin(np.linspace(0, 2, n_df)))
    rainfall = get_series(df, 'rainfall', np.random.randint(0, 20, n_df))
    wind_speed = get_series(df, 'wind_speed', 10 + np.random.normal(0, 2, n_df))

    n = min(len(dates), len(temperature), len(humidity), len(rainfall), len(wind_speed))
    if n < 1:
        raise ValueError("Dataset has no valid numeric rows.")
    dates = np.asarray(dates).ravel()[:n]
    temperature = np.asarray(temperature, dtype=float).ravel()[:n]
    humidity = np.asarray(humidity, dtype=float).ravel()[:n]
    rainfall = np.asarray(rainfall, dtype=float).ravel()[:n]
    wind_speed = np.asarray(wind_speed, dtype=float).ravel()[:n]

    if predicted_temp is None:
        if n >= 2:
            X = np.arange(n).reshape(-1, 1)
            model = LinearRegression().fit(X, temperature)
            predicted_temp = model.predict(X) + np.random.normal(0, 0.8, n)
        else:
            predicted_temp = temperature.copy()

    predicted_temp = np.asarray(predicted_temp, dtype=float).ravel()[:n]
    if len(predicted_temp) < n:
        predicted_temp = np.pad(predicted_temp, (0, n - len(predicted_temp)), mode='edge')

    min_len = n
    dates = dates[:min_len]
    temperature = temperature[:min_len]
    predicted_temp = predicted_temp[:min_len]
    humidity = humidity[:min_len]
    rainfall = rainfall[:min_len]
    wind_speed = wind_speed[:min_len]

    graphs = {}
    years_arr = None
    if 'date' in df.columns:
        try:
            years_arr = pd.to_datetime(dates, errors='coerce').year
        except Exception:
            years_arr = None
    try:
        dates_plt = pd.to_datetime(dates) if (hasattr(dates[0], 'isoformat') or (isinstance(dates[0], str) and len(str(dates[0])) > 8)) else dates
    except Exception:
        dates_plt = np.arange(min_len)

    if include('historical_temp'):
        fig, ax = plt.subplots(figsize=(8, 4))
        if years_arr is not None:
            valid_mask = ~pd.isna(years_arr)
            if valid_mask.any():
                df_hist = pd.DataFrame({
                    'year': years_arr[valid_mask].astype(int),
                    'temperature': temperature[valid_mask],
                })
                agg = df_hist.groupby('year')['temperature'].mean().sort_index()
                x_labels = agg.index.astype(int).astype(str).tolist()
                y_vals = agg.values
                ax.bar(x_labels, y_vals, color='#e74c3c', edgecolor='#c0392b')
                ax.set_xlabel("Year")
                plt.xticks(rotation=45)
            else:
                ax.bar(dates_plt, temperature, color='#e74c3c', edgecolor='#c0392b')
                ax.set_xlabel("Date")
        else:
            ax.bar(dates_plt, temperature, color='#e74c3c', edgecolor='#c0392b')
            ax.set_xlabel("Date")

        ax.set_title("Historical Temperature Trend ", fontsize=12)
        ax.set_ylabel("Temperature (°C)")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graphs['historical_temp'] = base64.b64encode(buf.getvalue()).decode()
        plt.close()

    if include('actual_vs_predicted'):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates_plt, temperature, label='Actual', color='#3498db', linewidth=2)
        ax.plot(dates_plt, predicted_temp, label='Predicted', color='#e67e22', linewidth=2, linestyle='--')
        ax.set_title("Actual vs Predicted Temperature", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graphs['actual_vs_predicted'] = base64.b64encode(buf.getvalue()).decode()
        plt.close()

    if include('humidity'):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(humidity, bins='auto', color='#9b59b6', edgecolor='#6c3483', alpha=0.85)
        ax.set_title("Humidity Trend ", fontsize=12)
        ax.set_xlabel("Humidity (%)")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graphs['humidity'] = base64.b64encode(buf.getvalue()).decode()
        plt.close()

    if include('rainfall'):
        # Build pyramid-style chart for rainfall intensity categories.
        low_rain = float(np.sum(rainfall < 2))
        mid_rain = float(np.sum((rainfall >= 2) & (rainfall < 10)))
        high_rain = float(np.sum((rainfall >= 10) & (rainfall < 25)))
        storm_rain = float(np.sum(rainfall >= 25))

        category_labels = ['Storm (>=25mm)', 'High (10-25mm)', 'Moderate (2-10mm)', 'Low (<2mm)']
        category_values = [max(1.0, storm_rain), max(1.0, high_rain), max(1.0, mid_rain), max(1.0, low_rain)]

        fig, ax = plt.subplots(figsize=(8, 6))

        y = np.arange(len(category_labels))
        ax.barh(y, category_values, color=['#e74c3c', '#f39c12', '#3498db', '#1abc9c'], edgecolor='black')
        ax.set_yticks(y)
        ax.set_yticklabels(category_labels)
        ax.invert_yaxis()

        max_val = max(category_values) if category_values else 1
        ax.set_xlim(0, max_val * 1.1)

        for i, v in enumerate(category_values):
            ax.text(v + max_val * 0.01, i, str(int(v)), va='center', fontweight='bold')

        ax.set_title('Rainfall Intensity Pyramid Chart', fontsize=14)
        ax.set_xlabel('Number of Days')
        ax.set_ylabel('Rainfall Category')

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graphs['rainfall'] = base64.b64encode(buf.getvalue()).decode()
        plt.close()

    if include('wind_speed'):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates_plt, wind_speed, color='#34495e', linewidth=2)
        ax.set_title("Wind Speed Variation", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Wind Speed (km/h)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graphs['wind_speed'] = base64.b64encode(buf.getvalue()).decode()
        plt.close()

    if include('distribution'):
        if 'condition' in df.columns:
            cond_counts = df['condition'].value_counts()
            conditions = cond_counts.index.tolist()
            counts = cond_counts.values.tolist()
        else:
            conditions = ["Sunny", "Cloudy", "Rainy", "Stormy"]
            low_rain = np.sum(rainfall < 2)
            mid_rain = np.sum((rainfall >= 2) & (rainfall < 10))
            high_rain = np.sum((rainfall >= 10) & (rainfall < 25))
            storm = np.sum(rainfall >= 25)
            counts = [max(1, low_rain), max(1, mid_rain), max(1, high_rain), max(1, storm)]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(counts, labels=conditions, autopct='%1.1f%%', colors=['#f1c40f', '#95a5a6', '#3498db', '#e74c3c'])
        ax.set_title("Weather Condition Distribution", fontsize=12)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graphs['distribution'] = base64.b64encode(buf.getvalue()).decode()
        plt.close()

    mae = mean_absolute_error(temperature, predicted_temp)
    rmse = np.sqrt(mean_squared_error(temperature, predicted_temp))
    if include('error_analysis'):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(temperature, predicted_temp, alpha=0.7, c='#3498db')
        min_val = min(temperature.min(), predicted_temp.min())
        max_val = max(temperature.max(), predicted_temp.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        ax.set_title("Actual vs Predicted (Error Analysis)", fontsize=12)
        ax.set_xlabel("Actual Temperature")
        ax.set_ylabel("Predicted Temperature")
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graphs['error_analysis'] = base64.b64encode(buf.getvalue()).decode()
        plt.close()

    metrics = {'mae': round(mae, 2), 'rmse': round(rmse, 2)}
    return graphs, metrics


# ---------- Routes (connect to templates) ----------

@app.route('/')
def index():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))
    return redirect(url_for('upload'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('admin_logged_in'):
        return redirect(url_for('upload'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if username == ADMIN_USER and password == ADMIN_PASS:
            session['admin_logged_in'] = True
            flash('Logged in successfully.', 'success')
            return redirect(url_for('upload'))
        users = load_users()
        if username in users and check_password_hash(users[username], password):
            session['admin_logged_in'] = True
            flash('Logged in successfully.', 'success')
            return redirect(url_for('upload'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')  # uses base.html


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if session.get('admin_logged_in'):
        return redirect(url_for('upload'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        if not username or not password:
            flash('Username and password are required.', 'error')
            return redirect(url_for('signup'))
        if len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
            return redirect(url_for('signup'))
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('signup'))
        if len(password) < 4:
            flash('Password must be at least 4 characters.', 'error')
            return redirect(url_for('signup'))
        users = load_users()
        if username in users:
            flash('Username already exists. Sign in or choose another.', 'error')
            return redirect(url_for('signup'))
        if username == ADMIN_USER:
            flash('This username is reserved.', 'error')
            return redirect(url_for('signup'))
        users[username] = generate_password_hash(password)
        save_users(users)
        flash('Account created. You can now sign in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        use_sample = request.form.get('use_sample_10') == 'yes'
        path = os.path.join(UPLOAD_FOLDER, 'current_dataset.csv')
        if use_sample:
            # Ensure the 10-year sample exists (auto-generated from legacy file if needed).
            if ensure_sample_10years() and os.path.isfile(SAMPLE_10YEARS):
                shutil.copy(SAMPLE_10YEARS, path)
                flash('Loaded sample 10-year dataset. Opening predictions...', 'success')
                return redirect(url_for('predict'))
            flash('Sample 10-year dataset is not available. Please upload a CSV file.', 'error')
            return redirect(url_for('upload'))
        f = request.files.get('dataset')
        if not f or f.filename == '':
            flash('No file selected.', 'error')
            return redirect(url_for('upload'))
        if not f.filename.lower().endswith('.csv'):
            flash('Please upload a CSV file.', 'error')
            return redirect(url_for('upload'))
        try:
            f.save(path)
        except Exception as e:
            flash(f'Could not save file: {e}', 'error')
            return redirect(url_for('upload'))
        df, err = parse_uploaded_csv(path)
        if err:
            flash(f'Invalid dataset: {err}', 'error')
            return redirect(url_for('upload'))
        flash('Dataset uploaded. Opening predictions and graphs...', 'success')
        return redirect(url_for('predict'))
    return render_template('upload.html')  # uses base.html


@app.route('/predict', methods=['GET'])
def predict():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))
    path = os.path.join(UPLOAD_FOLDER, 'current_dataset.csv')
    if not os.path.isfile(path):
        flash('Please upload a dataset first or use the sample 10-year dataset.', 'error')
        return redirect(url_for('upload'))
    df, err = parse_uploaded_csv(path)
    if err:
        flash(f'Error reading dataset: {err}', 'error')
        return redirect(url_for('upload'))
    df, years = filter_df_to_last_n_years(df, 10)
    selected_year = request.args.get('year', 'all')
    selected_graphs = request.args.getlist('graphs')
    if selected_year != 'all' and years and str(selected_year).isdigit():
        if int(selected_year) not in years:
            selected_year = 'all'
            flash('Selected year not in the last 10 years range. Showing all available years.', 'info')
    df_work = filter_df_by_year(df, selected_year if selected_year != 'all' else None)
    if len(df_work) < 3 and len(df) >= 3:
        flash('Too few rows for selected year. Showing all years.', 'info')
        df_work = df
        selected_year = 'all'
    elif len(df_work) < 1:
        flash('No data for selected year.', 'error')
        return redirect(url_for('upload'))
    try:
        graphs, metrics = generate_all_graphs(df_work, which_graphs=selected_graphs if selected_graphs else None)
        return render_template(
            'results.html',  # uses base.html
            graphs=graphs,
            metrics=metrics,
            graph_options=GRAPH_OPTIONS,
            selected_graphs=selected_graphs,
            years=years,
            selected_year=selected_year,
        )
    except Exception as e:
        flash(f'Run failed: {str(e)}. Check your CSV has date, temperature, humidity, rainfall, wind_speed.', 'error')
        return redirect(url_for('upload'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
