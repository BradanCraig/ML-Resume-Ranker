# ML Resume Ranker

A machine learning project that ranks resumes based on predicted job-match scores.

---

## Files

### `resume_data.csv`
Raw Kaggle dataset containing 9,000+ resumes and 35 columns of resume data.

### `main.py`
Preprocesses and cleans the dataset by:
- Handling `NaN` values
- Casting data types
- Removing unnecessary columns

### `dataset.csv`
Cleaned dataset used for model training.

### `model.py`
Trains the machine learning model, generates predictions, and creates performance visualizations.

### `top_5_applications.csv`
Contains the top 5 highest-ranked applications predicted by the model.

---