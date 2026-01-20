from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import joblib
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
app.secret_key = "secret_key"

# Directory to save uploaded avatars
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AVATAR_DIR = os.path.join(BASE_DIR, "static", "images", "avatars")
os.makedirs(AVATAR_DIR, exist_ok=True)

# File for local persistence
USERS_FILE = os.path.join(BASE_DIR, "users.json")

# ---------------- LOAD MODEL ----------------
# Adjust path to find the 'model' folder (sibling to 'interface')
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'model')

MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_Decision_Tree.pkl')
LE_FILIERE_PATH = os.path.join(MODEL_DIR, 'label_encoder_filiere.pkl')
LE_REC_PATH = os.path.join(MODEL_DIR, 'label_encoder_recommendation.pkl')

print("Loading AI Models...")
try:
    model = joblib.load(MODEL_PATH)
    le_filiere = joblib.load(LE_FILIERE_PATH)
    le_rec = joblib.load(LE_REC_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def predict_orientation(moyenne, maths, physique, francais, filiere):
    if model is None:
        return "Model Unavailable"
    
    try:
        # Create DataFrame
        input_data = pd.DataFrame({
            'Moyenne_Generale': [float(moyenne)],
            'Note_Maths': [float(maths)],
            'Note_Physique': [float(physique)],
            'Note_Francais': [float(francais)],
            'Filiere_Bac': [filiere]
        })
        
        # Encode Filiere
        input_data['Filiere_Bac_Encoded'] = le_filiere.transform(input_data['Filiere_Bac'])
        
        # Select Features
        features = ['Moyenne_Generale', 'Note_Maths', 'Note_Physique', 'Note_Francais', 'Filiere_Bac_Encoded']
        X = input_data[features]
        
        # Predict
        pred_encoded = model.predict(X)
        recommendation = le_rec.inverse_transform(pred_encoded)[0]
        return recommendation
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Error in Prediction"

# -----------------------------
# User Storage with Persistence
# -----------------------------
def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading users: {e}")
            return {}
    return {}

def save_users(data):
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving users: {e}")

users = load_users()

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username in users and users[username]["password"] == password:
            session["username"] = username
            session["email"] = users[username]["email"]
            session["avatar"] = users[username].get("avatar", url_for('static', filename='images/default_avatar.png'))
            flash("Login successful!", "success")
            return redirect(url_for("profile"))
        else:
            flash("Invalid username or password!", "error")
            return redirect(url_for("login"))

    return render_template("login.html")


# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        # Validate fields
        if not username or not email or not password:
            flash("Please fill in all fields!", "error")
            return redirect(url_for("signup"))

        if username in users:
            flash("Username already exists!", "error")
            return redirect(url_for("signup"))

        # Save user including avatar
        users[username] = {
            "email": email,
            "password": password,
            "avatar": url_for('static', filename='images/default_avatar.png')
        }
        
        # Save to local file
        save_users(users)

        session["username"] = username
        session["email"] = email
        session["avatar"] = users[username]["avatar"]

        flash("Signup successful!", "success")
        return redirect(url_for("profile"))

    return render_template("signup.html")


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully!", "success")
    return redirect(url_for("login"))


# ---------------- PROFILE ----------------
@app.route("/", methods=["GET", "POST"])
def profile():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        student_form = {
            "bac_mean": request.form.get("bac_mean"),
            "math_grade": request.form.get("math_grade"),
            "pc_grade": request.form.get("pc_grade"),
            "fr_grade": request.form.get("fr_grade"),
            "bac_field": request.form.get("bac_field")
        }

        if not all(student_form.values()):
            flash("Please fill in all fields!", "error")
            return redirect(url_for("profile"))

        # --- SERVER-SIDE VALIDATION ---
        try:
            # 1. Validate Grades (0-20)
            grades = [
                float(student_form["bac_mean"]),
                float(student_form["math_grade"]),
                float(student_form["pc_grade"]),
                float(student_form["fr_grade"])
            ]
            if any(g < 0 or g > 20 for g in grades):
                flash("All grades must be between 0 and 20!", "error")
                return redirect(url_for("profile"))

            # 2. Validate BAC Field
            allowed_fields = ['SM-A', 'SM-B', 'PC', 'SVT', 'Eco-Gestion']
            if student_form["bac_field"] not in allowed_fields:
                flash(f"Invalid BAC field! Allowed: {', '.join(allowed_fields)}", "error")
                return redirect(url_for("profile"))

        except ValueError:
            flash("Invalid number format provided!", "error")
            return redirect(url_for("profile"))

        session["student"] = student_form
        
        # --- AI PREDICTION ---
        ai_rec = predict_orientation(
            student_form["bac_mean"],
            student_form["math_grade"],
            student_form["pc_grade"],
            student_form["fr_grade"],
            student_form["bac_field"]
        )
        session["ai_recommendation"] = ai_rec
        
        flash("Profile saved successfully!", "success")
        return redirect(url_for("recommendations"))

    student = session.get("student")
    return render_template("profile.html", student=student)


# ---------------- UPDATE PROFILE ----------------
@app.route("/update_profile", methods=["POST"])
def update_profile():
    if "username" not in session:
        return redirect(url_for("login"))

    current_username = session["username"]

    # Get new data
    new_username = request.form.get("username")
    new_email = request.form.get("email")
    new_password = request.form.get("password")
    avatar_file = request.files.get("avatar")

    # Fetch user data safely, or create new
    user_data = users.get(current_username, {
        "email": session.get("email", ""),
        "password": "",
        "avatar": session.get("avatar", url_for('static', filename='images/default_avatar.png'))
    })

    # Update data
    user_data["email"] = new_email
    if new_password:
        user_data["password"] = new_password

    # Handle avatar upload
    if avatar_file and avatar_file.filename != "":
        filename = f"{new_username}.png"
        filepath = os.path.join(AVATAR_DIR, filename)
        try:
            avatar_file.save(filepath)
            # Save the URL path relative to static
            user_data["avatar"] = url_for('static', filename=f'images/avatars/{filename}')
        except Exception as e:
            print(f"Failed to save avatar: {e}")

    # Remove old username if changed
    if new_username != current_username:
        users.pop(current_username, None)

    users[new_username] = user_data
    save_users(users)  # Persist changes

    # Update session
    session["username"] = new_username
    session["email"] = new_email
    session["avatar"] = user_data["avatar"]

    flash("Profile updated successfully!", "success")
    return redirect(url_for("profile"))


# ---------------- RECOMMENDATIONS ----------------
@app.route("/recommendations", methods=["GET", "POST"])
def recommendations():
    if "username" not in session:
        return redirect(url_for("login"))

    if "student" not in session:
        return redirect(url_for("profile"))

    if request.method == "POST":
        # Use the stored AI recommendation
        rec = session.get("ai_recommendation", "General Orientation")
        session["recommendations"] = [
            f"Top Recommendation: {rec}",
            f"Alternative: Check {rec} requirements",
            "General University Guidance"
        ]
        return redirect(url_for("dashboard"))

    return render_template("recommendations.html", student=session["student"], ai_recommendation=session.get("ai_recommendation"))


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    if "student" not in session or "recommendations" not in session:
        return redirect(url_for("profile"))

    return render_template(
        "dashboard.html",
        student=session["student"],
        recommendations=session["recommendations"]
    )


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
