import sys
import os

from werkzeug.utils import secure_filename

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import cv2
from flask import Flask, render_template, session, url_for
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import re  # For input validation
import os
from flask import request, flash, redirect
from flask import send_from_directory
from predict_image import predict_image
from segmentation import process_image

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

UPLOAD_FOLDER = '/Users/ecekocabay/Desktop/CNG491/BreastCancerGUI 2/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.getenv('MODEL_PATH', '/Users/ecekocabay/Desktop/CNG491/BreastCancerGUI 2/random_forest_model.pkl')

from functools import wraps

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

        # Validate required fields
        if not fullname:
            flash('Full Name is required.', 'danger')
            return render_template('register.html')
        if not diploma_number:
            flash('Diploma Number is required.', 'danger')
            return render_template('register.html')
        if not email:
            flash('Email is required.', 'danger')
            return render_template('register.html')
        if not password:
            flash('Password is required.', 'danger')
            return render_template('register.html')

        # Validate password strength
        password_regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d]{8,}$'
        if not re.match(password_regex, password):
            flash('Password should be at least 8 characters long, include one uppercase letter, one lowercase letter, and one number.', 'danger')
            return render_template('register.html')

        # Check for duplicate diploma number
        conn = sqlite3.connect('radiologist_system.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM Radiologists WHERE diploma_number = ?', (diploma_number,))
        duplicate_diploma = cursor.fetchone()

        if duplicate_diploma:
            flash('There is already someone with this diploma number.', 'danger')
            conn.close()
            return render_template('register.html')

        # Hash the password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Insert into database
        try:
            cursor.execute('''
                INSERT INTO Radiologists (fullname, diploma_number, email, password)
                VALUES (?, ?, ?, ?)
            ''', (fullname, diploma_number, email, hashed_password))
            conn.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect('/')  # Redirect to the login page
        except sqlite3.IntegrityError as e:
            flash(f'An unexpected error occurred during registration: {str(e)}', 'danger')
        finally:
            conn.close()

    return render_template('register.html')  # Render the registration page on GET requests or errors
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        diploma_number = request.form.get('diploma_number', '').strip()
        password = request.form.get('password', '').strip()

        # Validate required fields
        if not diploma_number:
            flash('Diploma Number is required.', 'danger')
            return render_template('login.html')
        if not password:
            flash('Password is required.', 'danger')
            return render_template('login.html')

        # Check if the diploma number exists in the database
        conn = sqlite3.connect('radiologist_system.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, password FROM Radiologists WHERE diploma_number = ?
        ''', (diploma_number,))
        radiologist = cursor.fetchone()
        conn.close()

        if not radiologist:
            # Diploma number not found
            flash('No account found with the provided diploma number.', 'danger')
            return render_template('login.html')

        # Check if the password matches
        radiologist_id, hashed_password = radiologist
        if not check_password_hash(hashed_password, password):
            # Password incorrect
            flash('Incorrect password. Please try again.', 'danger')
            return render_template('login.html')

        # Store radiologist ID in session
        session['user_id'] = radiologist_id

        # Successful login
        flash('Login successful!', 'success')
        return redirect('/home')  # Redirect to the home page

    return render_template('login.html')

@app.route('/home')
@login_required
def home():
    return render_template('home.html')
@app.route('/')
def index():
    session.clear()
    return render_template('index.html')  # Render the homepage


@app.route('/view_previous', methods=['GET', 'POST'])
@login_required
def view_previous():
    radiologist_id = session.get('user_id')
    if not radiologist_id:
        flash('Session expired. Please log in again.', 'danger')
        return redirect('/')

    conn = sqlite3.connect('radiologist_system.db')
    cursor = conn.cursor()

    # Handle patient name filter if provided
    patient_name = request.form.get('patient_name', '').strip() if request.method == 'POST' else ''
    if patient_name:
        # Query for filtering by patient name
        cursor.execute('''
            SELECT CR.classification_date, CR.prediction, CR.confidence, P.fullname AS patient_name
            FROM ClassificationResults CR
            JOIN Patients P ON CR.patient_id = P.id
            WHERE CR.radiologist_id = ? AND P.fullname LIKE ?
            ORDER BY CR.classification_date DESC
        ''', (radiologist_id, f"%{patient_name}%"))
    else:
        # Query for all results of the logged-in radiologist
        cursor.execute('''
            SELECT CR.classification_date, CR.prediction, CR.confidence, P.fullname AS patient_name
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
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"Debug: Original file saved at {file_path}")

        # Segment the image
        segmented_image = process_image(file_path)
        if segmented_image is None:
            flash('Segmentation failed. No region of interest detected.', 'danger')
            return redirect('/home')

        segmented_image_filename = f"segmented_{filename.split('.')[0]}.jpg"
        segmented_image_path = os.path.join(app.config['UPLOAD_FOLDER'], segmented_image_filename)
        cv2.imwrite(segmented_image_path, segmented_image)
        print(f"Debug: Segmented image saved at {segmented_image_path}")

        segmented_image_url = url_for('static', filename=f"uploads/{segmented_image_filename}")
        print(f"Debug: Segmented image URL: {segmented_image_url}")

        # Predict the class and confidence
        print("Debug: Starting prediction...")
        prediction, confidence = predict_image(segmented_image_path, MODEL_PATH)
        print(f"Debug: Prediction successful - Class: {prediction}, Confidence: {confidence}")

        # Render the result page
        return render_template(
            'result.html',
            segmented_image=segmented_image_url,
            prediction=prediction,
            confidence=confidence
        )
    except Exception as e:
        print(f"Debug: Error during processing - {str(e)}")
        flash(f"Error during processing: {str(e)}", 'danger')
        return redirect('/home')
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect('/')
if __name__ == "__main__":
    app.run(debug=True)