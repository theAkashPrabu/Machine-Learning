# Machine Learning with Titanic Dataset

This project uses the Titanic dataset to demonstrate data preprocessing, exploratory data analysis (EDA), and machine learning using Python. It employs a Random Forest Classifier to predict survival outcomes.

## Features
- Data preprocessing:
  - Handling missing values
  - Encoding categorical variables
  - Feature engineering (e.g., family size)
- Exploratory Data Analysis (EDA)
- Training a Random Forest Classifier
- Model evaluation using metrics such as confusion matrix, classification report, and ROC-AUC

## Requirements
The following Python libraries are used:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

Install them using pip if not already installed:
bash
pip install pandas numpy seaborn matplotlib scikit-learn


## How to Use
1. Clone the repository:
   bash
   git clone <repository-url>
   cd <repository-directory>
   

2. Run the Jupyter Notebook:
   bash
   jupyter notebook "ML AKASH.ipynb"
   

3. Follow the steps in the notebook to:
   - Load and preprocess the Titanic dataset
   - Train the machine learning model
   - Evaluate the model’s performance

## Dataset
The dataset used is the Titanic dataset, loaded using Seaborn:
python
sns.load_dataset('titanic')

Ensure that the dataset is accessible through Seaborn, or download it manually from [Kaggle](https://www.kaggle.com/c/titanic/data).

## Output
The notebook outputs:
- Preprocessed data
- Visualizations for EDA
- Model performance metrics
