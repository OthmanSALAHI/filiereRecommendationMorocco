# Student Orientation Recommendation System (Morocco)

This project is a Machine Learning application designed to recommend higher education paths (filières) for Moroccan high school students based on their Baccalaureate performance and specialization.

## 📌 Project Overview

The system uses a supervised learning approach to predict the most suitable higher education track for a student. It works by:
1.  Generating a synthetic dataset representing student profiles (grades and baccalaureate branch).
2.  Applying specific business logic (Moroccan orientation rules) to label the data.
3.  Training multiple Machine Learning models to learn these patterns.
4.  Selecting and saving the best performing model for future predictions.

## 📂 Project Structure

```
├── README.md               # Project documentation
├── data_training/          # Scripts for data generation and model training
│   ├── generate_data.py    # Script to generate synthetic dataset
│   ├── creating_modele.ipynb # Jupyter notebook for training and evaluating models
│   └── dataset_orientation_maroc_6000.csv # Generated dataset
├── interface/              # Web application
│   ├── app.py              # Main Flask application
│   ├── templates/          # HTML templates
│   ├── static/             # CSS and images
│   ├── users.json          # User data storage
│   └── requirements.txt    # Web app dependencies
└── model/                  # Saved artifacts
    ├── best_model_Decision_Tree.pkl # Best trained model
    ├── label_encoder_filiere.pkl    # Encoder for Bac branches
    └── label_encoder_recommendation.pkl # Encoder for output recommendations
```

## 🛠 Prerequisites

You need Python installed along with the following libraries (see `interface/requirements.txt`):

```bash
pip install pandas numpy scikit-learn matplotlib joblib flask
```

## 🚀 How to Use

### 1. Data Generation
To create the synthetic dataset used for training, run the generation script:

```bash
python data_training/generate_data.py
```
This will create `dataset_orientation_maroc_6000.csv` containing 6000 student profiles with features like:
- `Moyenne_Generale` (General Average)
- `Note_Maths`, `Note_Physique`, `Note_Francais`
- `Filiere_Bac` (High School Branch: SM-A, SM-B, PC, SVT, Eco-Gestion)
- `Recommendation` (Target Label: Medecine, CPGE, ENSA, ENCG, etc.)

### 2. Model Training & Evaluation
Open the notebook `data_training/creating_modele.ipynb`. This notebook performs the following:
- Loads the dataset.
- Visualizes data distributions.
- Preprocesses data (Label Encoding).
- Trains 6 different algorithms:
    - Random Forest
    - Decision Tree
    - K-Nearest Neighbors (KNN)
    - Logistic Regression
    - Gradient Boosting
    - SVM
- Compares accuracy and saves the best model (e.g., Decision Tree) to the `model/` folder.

### 3. Web Interface
To run the web application for student orientation:

1. Navigate to the `interface` directory:
   ```bash
   cd interface
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. Open your browser and go to `http://127.0.0.1:5000` to access the application.
   - **Login/Signup**: Create an account or log in.
   - **Profile**: Enter your Baccalaureate grades and stream.
   - **Recommendation**: View the AI-predicted educational path.

## 📊 Methodology

### Business Logic (Data Labeling)
The synthetic data is labeled based on realistic orientation rules in Morocco:
- **Medecine / Pharmacie**: High general average (>16) and strong science scores.
- **CPGE (Prep School)**: Strong Maths (>15) and good general average.
- **ENSA / ENSAM**: Good Maths and Physics (>13).
- **ENCG**: Eco-Gestion students with good grades.
- **FST / Fac**: Other profiles based on varying thresholds.

### Model Performance
The models are evaluated using Accuracy metrics and Classification Reports. The system automatically selects the model with the highest accuracy on the test set.

## 👥 Authors

- **Othman SALAHI**
- **Mohamed MAKRANI**
- **Malak HOUALI**