import atexit
import os
import sys
import uuid
import sqlite3
import re
from datetime import datetime

import cv2
from flask import Flask, render_template, session, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from predict_ml import MLClassifier
from segmentation import segment_image

# Flask app configuration
app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = '/Users/ecekocabay/Desktop/CNG491/BreastCancerGUI 2/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model and scaler paths
MODEL_PATH = "/Users/ecekocabay/Desktop/CNG491/BreastCancerGUI 2/models/random_forest_model.pkl"
SCALER_PATH = "/Users/ecekocabay/Desktop/CNG491/BreastCancerGUI 2/process/scaler.pkl"

# Initialize ML Classifier
ml_classifier = MLClassifier(MODEL_PATH, SCALER_PATH)

# Cleanup function to delete all files in the upload folder
def cleanup_upload_folder():
    print("Cleaning up upload folder...")
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Remove the file
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

# Register the cleanup function to run at exit
atexit.register(cleanup_upload_folder)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'danger')
            return redirect('/')
        return f(*args, **kwargs)
    return decorated_function


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form.get('fullname', '').strip()
        diploma_number = request.form.get('diploma_number', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not fullname or not diploma_number or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('register.html')

        password_regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d]{8,}$'
        if not re.match(password_regex, password):
            flash('Password must be 8+ characters long, including an uppercase, lowercase, and a number.', 'danger')
            return render_template('register.html')

        conn = sqlite3.connect('radiologist_system.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM Radiologists WHERE diploma_number = ?', (diploma_number,))
        if cursor.fetchone():
            flash('This diploma number is already registered.', 'danger')
            conn.close()
            return render_template('register.html')

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        try:
            cursor.execute('''
                INSERT INTO Radiologists (fullname, diploma_number, email, password)
                VALUES (?, ?, ?, ?)
            ''', (fullname, diploma_number, email, hashed_password))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect('/')
        except sqlite3.IntegrityError as e:
            flash(f'Error during registration: {str(e)}', 'danger')
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        diploma_number = request.form.get('diploma_number', '').strip()
        password = request.form.get('password', '').strip()

        if not diploma_number or not password:
            return render_template('login.html', error_message="Both fields are required.")

        conn = sqlite3.connect('radiologist_system.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, password FROM Radiologists WHERE diploma_number = ?', (diploma_number,))
        radiologist = cursor.fetchone()
        conn.close()

        if not radiologist or not check_password_hash(radiologist[1], password):
            return render_template('login.html', error_message="Invalid Diploma Number or Password.")

        # Login success
        session['user_id'] = radiologist[0]
        return redirect('/home')

    return render_template('login.html')
@app.route('/home')
@login_required
def home():
    return render_template('home.html')


@app.route('/')
def index():
    session.clear()
    return render_template('index.html')


@app.route('/view_previous', methods=['GET', 'POST'])
@login_required
def view_previous():
    radiologist_id = session.get('user_id')
    if not radiologist_id:
        flash('Session expired. Please log in again.', 'danger')
        return redirect('/')

    conn = sqlite3.connect('radiologist_system.db')
    cursor = conn.cursor()

    patient_name = request.form.get('patient_name', '').strip() if request.method == 'POST' else ''
    if patient_name:
        cursor.execute('''
            SELECT CR.classification_date, CR.prediction, CR.confidence, P.fullname
            FROM ClassificationResults CR
            JOIN Patients P ON CR.patient_id = P.id
            WHERE CR.radiologist_id = ? AND P.fullname LIKE ?
            ORDER BY CR.classification_date DESC
        ''', (radiologist_id, f"%{patient_name}%"))
    else:
        cursor.execute('''
            SELECT CR.classification_date, CR.prediction, CR.confidence, P.fullname
            FROM ClassificationResults CR
            JOIN Patients P ON CR.patient_id = P.id
            WHERE CR.radiologist_id = ?
            ORDER BY CR.classification_date DESC
        ''', (radiologist_id,))

    results = cursor.fetchall()
    conn.close()
    return render_template('previous_results.html', results=results, patient_name=patient_name)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process', methods=['POST'])
@login_required
def process():
    patient_name = request.form.get('patient_name', '').strip()
    if not patient_name:
        flash('Patient name is required.', 'danger')
        return redirect('/home')

    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect('/home')

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect('/home')

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"Original uploaded file path: {file_path}")

        # Perform prediction (segmentation handled inside the classifier)
        predicted_class, confidence, segmented_image_path = ml_classifier.predict(file_path)

        # Generate URL for segmented image
        segmented_image_filename = os.path.basename(segmented_image_path)
        segmented_image_url = url_for('uploaded_file', filename=segmented_image_filename)
        print(f"Generated URL for segmented image: {segmented_image_url}")

        # Store results in the database
        conn = sqlite3.connect('radiologist_system.db')
        cursor = conn.cursor()

        # Get the radiologist's ID from the session
        radiologist_id = session.get('user_id')

        # Check if the patient already exists in the database
        cursor.execute('SELECT id FROM Patients WHERE fullname = ?', (patient_name,))
        patient = cursor.fetchone()

        if not patient:
            # Insert new patient if not found
            current_date = datetime.now().strftime('%Y-%m-%d')
            cursor.execute(
                '''
                INSERT INTO Patients (fullname, radiologist_id, dob, patient_id)
                VALUES (?, ?, ?, ?)
                ''',
                (patient_name, radiologist_id, current_date, str(uuid.uuid4()))
            )
            conn.commit()
            patient_id = cursor.lastrowid
        else:
            # Use existing patient's ID
            patient_id = patient[0]

        # Insert classification result into the database
        cursor.execute(
            '''
            INSERT INTO ClassificationResults (patient_id, radiologist_id, prediction, confidence, classification_date)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (patient_id, radiologist_id, predicted_class, confidence, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        conn.commit()
        conn.close()

        # Render result
        return render_template(
            'result.html',
            segmented_image=segmented_image_url,
            prediction=predicted_class,
            confidence=confidence
        )
    except Exception as e:
        print(f"Debug: Error during processing - {str(e)}")
        flash(f"Error during processing: {str(e)}", 'danger')
        return redirect('/home')
@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect('/')


if __name__ == "__main__":
    app.run(debug=True)