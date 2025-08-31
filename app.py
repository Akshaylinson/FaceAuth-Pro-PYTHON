from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from base64 import b64decode
from deepface import DeepFace
import os
import time
from datetime import timedelta
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")
app.permanent_session_lifetime = timedelta(minutes=30)

# ✅ Allow bigger uploads (16 MB limit)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STUDENTS_DIR = "students"
os.makedirs(STUDENTS_DIR, exist_ok=True)

# Configuration
MODEL_NAME = "Facenet512"
DETECTOR = "opencv"
MAX_FILE_AGE_DAYS = 30  # Clean up files older than this


def save_base64_image(image_data_b64: str, path: str) -> None:
    """Save a base64 dataURL image to disk."""
    try:
        header, encoded = image_data_b64.split(",", 1)
    except ValueError:
        raise ValueError("Invalid image data format")
    with open(path, "wb") as f:
        f.write(b64decode(encoded))


def sanitize_email(email: str) -> str:
    """Create a safe filename from an email."""
    return email.strip().lower().replace("@", "_at_").replace(".", "_")


def single_face_present(img_path: str) -> bool:
    """Return True if exactly one face is detected."""
    try:
        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=DETECTOR,
            enforce_detection=True
        )
        return len(faces) == 1
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return False


def cleanup_old_files():
    """Remove temporary files older than MAX_FILE_AGE_DAYS"""
    try:
        now = time.time()
        for filename in os.listdir(STUDENTS_DIR):
            if filename.startswith("temp_"):
                filepath = os.path.join(STUDENTS_DIR, filename)
                if os.path.getmtime(filepath) < now - MAX_FILE_AGE_DAYS * 86400:
                    os.remove(filepath)
                    logger.info(f"Removed old temp file: {filename}")
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")


@app.route("/", methods=["GET"])
def home():
    # Clean up old files on home page load
    cleanup_old_files()
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    mode = request.form.get("mode")  # "login" or "register"
    email = (request.form.get("email") or "").strip().lower()
    image_data = request.form.get("image_data") or ""

    if not email or mode not in ("login", "register") or not image_data:
        flash("Missing email, image, or mode.", "error")
        return redirect(url_for("home"))

    fname = sanitize_email(email)
    registered_path = os.path.join(STUDENTS_DIR, f"{fname}.jpg")
    temp_path = os.path.join(STUDENTS_DIR, f"temp_{fname}_{os.urandom(4).hex()}.jpg")

    try:
        save_base64_image(image_data, temp_path)
    except Exception as e:
        flash(f"Could not decode captured image: {e}", "error")
        return redirect(url_for("home"))

    try:
        # Ensure captured photo has exactly one face
        if not single_face_present(temp_path):
            flash("No face or multiple faces detected. Please recapture with your face centered and well-lit.", "error")
            return redirect(url_for("home"))

        if mode == "register":
            if os.path.exists(registered_path):
                flash("You're already registered with this email. Try logging in.", "info")
                return redirect(url_for("home"))

            os.replace(temp_path, registered_path)
            flash("Registration successful! You are now logged in.", "success")
            session.permanent = True
            session['email'] = email
            session['logged_in'] = True
            return redirect(url_for("dashboard"))  # ✅ Go to dashboard

        # mode == "login"
        if not os.path.exists(registered_path):
            flash("No registered face found for this email. Please register first.", "error")
            return redirect(url_for("home"))

        # 1:1 face verification
        result = DeepFace.verify(
            img1_path=temp_path,
            img2_path=registered_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR
        )

        if bool(result.get("verified")):
            flash(f"Welcome back, {email}!", "success")
            session.permanent = True
            session['email'] = email
            session['logged_in'] = True
            return redirect(url_for("dashboard"))  # ✅ Go to dashboard
        else:
            distance = result.get('distance', 1)
            similarity = max(0, (1 - distance) * 100)
            flash(f"Face not recognized (similarity: {similarity:.1f}%). Try again with better lighting and a neutral expression.", "error")

        return redirect(url_for("home"))

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        flash(f"Authentication error: {e}", "error")
        return redirect(url_for("home"))
    finally:
        # Best-effort cleanup of the temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        flash("You need to log in first.", "error")
        return redirect(url_for("home"))
    return render_template("dashboard.html", email=session.get("email"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("home"))


@app.route("/check_email", methods=["POST"])
def check_email():
    """Check if email is already registered"""
    email = (request.form.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "Email required"}), 400

    fname = sanitize_email(email)
    registered_path = os.path.join(STUDENTS_DIR, f"{fname}.jpg")

    return jsonify({"registered": os.path.exists(registered_path)})


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "False").lower() == "true")
