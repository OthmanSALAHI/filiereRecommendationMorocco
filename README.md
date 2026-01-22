# Student Orientation Recommendation System (Morocco)

This project is a Machine Learning application designed to recommend higher education paths (filières) for Moroccan high school students based on their Baccalaureate performance and specialization.

## 📌 Project Overview

The system uses a supervised learning approach to predict the most suitable higher education track for a student. It works by:
1.  Generating a synthetic dataset representing student profiles (grades and baccalaureate branch).
2.  Applying specific business logic (Moroccan orientation rules) to label the data.
3.  Training multiple Machine Learning models to learn these patterns.
4.  Providing a web interface for students to get real-time recommendations and for admins to monitor model performance.

## 📂 Project Structure

```
├── README.md               # Project documentation
├── data_training/          # Scripts for data generation and model training
│   ├── generate_data.py    # Script to generate synthetic dataset
│   ├── creating_modele.ipynb # Jupyter notebook for offline training
│   └── dataset_orientation_maroc_6000.csv # Generated dataset
├── interface/              # Web application
│   ├── app.py              # Main Flask application
│   ├── users.json          # User data storage (JSON based)
│   ├── templates/          # HTML templates (includes admin.html)
│   ├── static/             # CSS, images, and user avatars
│   └── requirements.txt    # Web app dependencies
└── model/                  # Saved artifacts
    ├── best_model_Decision_Tree.pkl # Best trained model
    ├── label_encoder_filiere.pkl    # Encoder for Bac branches
    └── label_encoder_recommendation.pkl # Encoder for output recommendations
```

## 🛠 Prerequisites

You need Python installed along with the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib joblib flask
```

## 🚀 How to Use

### 1. Data Generation
To create the synthetic dataset used for training, run the generation script:

```bash
python data_training/generate_data.py
```
This will create `dataset_orientation_maroc_6000.csv` containing 6000 student profiles.

### 2. Model Training (Offline)
Open `data_training/creating_modele.ipynb` to train models, visualize data distributions, and save the best performing model to the `model/` folder.

### 3. Web Application
To run the full student orientation platform:

1. Navigate to the `interface` directory:
   ```bash
   cd interface
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. Open your browser at `http://127.0.0.1:5000`.

## ✨ New Features

### 🎓 Student Portal
- **Smart Prediction**: Enter Baccalaureate grades (Math, Physics, French) and stream to get an instant recommendation.
- **User Accounts**: Secure Login and Signup system.
- **Profile Customization**: Users can update their profile and upload **custom avatars**.

### 🛡️ Admin Dashboard
The application now includes a powerful **Admin Dashboard** for monitoring AI performance.

- **Access**: Sign up or log in with the username **`root`**.
- **Real-time Evaluation**: The dashboard loads the training dataset and trains 6 different models on the fly (Decision Tree, Random Forest, KNN, SVM, etc.).
- **Visual Analytics**: Displays a comparative bar chart and detailed accuracy tables to check which model performs best on the current data.

## 📊 Methodology

### Business Logic
The dataset is labeled based on Moroccan orientation rules:
- **Medecine**: High average (>16) + strong science scores.
- **CPGE**: Strong Math (>15).
- **ENSA/ENSAM**: Good Math & Physics.
- **ENCG**: Focus on Eco-Gestion backgrounds.

### Model Comparison
The admin page compares the following algorithms:
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Gradient Boosting

## 👥 Authors

- **Othman SALAHI**
- **Mohamed MAKRANI**
- **Malak HOUALI**