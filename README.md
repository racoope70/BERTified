# Sentiment Analysis with BERT

**Table of Contents**
1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Project Structure](#project-structure)  
5. [Usage](#usage)  
6. [Model Performance](#model-performance)  
7. [Real-World Potential & Tangible Benefits](#real-world-potential--tangible-benefits)  
8. [Further Innovations & Expansion Plans](#further-innovations--expansion-plans)  
9. [License](#license)  
10. [Acknowledgments](#acknowledgments)

---

## Overview
This project leverages the **BERT** model to perform sentiment analysis on the **IMDb** dataset. By processing text data, tokenizing it with BERT’s tokenizer, and fine-tuning on labeled sentiment data, the model classifies movie reviews as either positive or negative—achieving a **91%** test accuracy. Recent updates address class imbalance via **SMOTE**, while comprehensive evaluation uses advanced metrics and visualizations to assess model performance.

---

## Features
- **Data Preprocessing**  
  - Cleans and tokenizes raw movie reviews for BERT compatibility.  
- **Model Training**  
  - Fine-tunes a pre-trained BERT model on the labeled sentiment data.  
- **Class Imbalance Handling**  
  - Employs **SMOTE** (Synthetic Minority Oversampling Technique) to mitigate skewed class distributions.  
- **Advanced Metrics & Visualizations**  
  - Analyzes performance using metrics such as **confusion matrix**, **ROC-AUC**, **precision**, **recall**, **F1-score**, and visual plots.

---

## Installation
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/sentiment-analysis-bert.git

2. **Navigate to the Project Directory**  
   ```bash
   cd sentiment-analysis-bert

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
Make sure you have Python 3.7+ installed.

sentiment-analysis-bert/
|
├── data/                      # IMDb dataset (if provided)
├── src/                       # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
|
├── notebooks/                 # Jupyter Notebooks
│   ├── IMDB_Sentiment_Analysis_with_BERT.ipynb
│   ├── Results_Rebalance_Dataset.ipynb
|
├── images/                    # Images for visualizations/plots
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
|
├── requirements.txt           # List of dependencies
├── LICENSE                    # License file
├── README.md                  # Project overview (this file)

**Usage**
**Option 1:** Jupyter Notebook
  - Open notebooks/IMDB_Sentiment_Analysis_with_BERT.ipynb.
  - Follow the cells in order to load data, preprocess, train the model, and evaluate performance.
**Option 2:** Command Line Scripts
1. **Prepare Data**
  - Place the IMDb dataset in the data/ folder (ensure correct structure, e.g., aclImdb/train and aclImdb/test).
    
2. **Data Preprocessing**  
   ```bash
   python src/data_preprocessing.py

3. **Train the Model**  
   ```bash
   python src/model_training.py
   
4. **Evaluate Performance**  
   ```bash
   python src/evaluation.py

**Model Performance**
**Training Accuracy:** ~93%
**Test Accuracy:** ~91%
**Confusion Matrix**

****True Negatives:** **1065
**False Positives:** 112
False Negatives: 71
True Positives: 752
**Key Metrics**

ROC-AUC: ~0.97 (strong class separation)
Precision: 0.87
Recall: 0.91
F1-Score: 0.89
Overall, these results confirm robust sentiment classification performance, balancing the detection of true positives with minimizing false alerts.

