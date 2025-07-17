
# Detection of Employee Stress Using Machine Learning

This project aims to detect **employee stress levels** using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. It processes social media texts (tweets) to classify whether an employee is under stress or not using classifiers like **Support Vector Machine (SVM)** and **Random Forest (RF)**.

## 🔍 Objective

To develop a stress detection system that analyzes employees' social media (e.g., Twitter) text data to identify stress patterns. This can be helpful in improving mental health awareness and workplace productivity.

---

## 📁 Project Structure

```text
Detection-of-Employee-stress-using-ML/
│
├── stress.csv                  # Dataset with text and stress labels
├── data_preprocessing.py       # Text cleaning, tokenization, stopword removal
├── feature_extraction.py       # TF-IDF vectorization
├── svm_model.py                # SVM model training & evaluation
├── random_forest_model.py      # Random Forest model training & evaluation
├── model_comparison.py         # Accuracy & metrics comparison
├── requirements.txt            # Required libraries
└── README.md                   # Project documentation
```

---

## 🧠 Machine Learning Models Used

- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

These models are trained on the preprocessed and vectorized text data to classify tweets as "stress" or "no stress".

---

## 📊 Dataset

- File: `stress.csv`
- Columns: `text`, `label`
- The dataset contains employee tweets or text data labeled with stress (1) or no stress (0).

---

## ⚙️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/DSSKalyan2004/Detection-of-Employee-stress-using-ML.git
cd Detection-of-Employee-stress-using-ML
```

### 2. Install required libraries
```bash
pip install -r requirements.txt
```

### 3. Preprocess the data
```bash
python data_preprocessing.py
```

### 4. Extract Features (TF-IDF)
```bash
python feature_extraction.py
```

### 5. Train and Test the SVM model
```bash
python svm_model.py
```

### 6. Train and Test the Random Forest model
```bash
python random_forest_model.py
```

### 7. Compare model performance
```bash
python model_comparison.py
```

---

## 🧪 Output

- Accuracy of each model
- Precision, Recall, F1-Score
- Confusion Matrix

---

## 📌 Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy
- NLTK
- Matplotlib, Seaborn
- Machine Learning & NLP

---

## 🙋‍♂️ Author

- **Daram Shiva Sai Kalyan**
- GitHub: [DSSKalyan2004](https://github.com/DSSKalyan2004)

---

## 📃 License

This project is open-source and available under the MIT License.
