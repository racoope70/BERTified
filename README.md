# Sentiment Analysis with BERT

**Table of Contents**
1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Project Structure](#project-structure)  
5. [Usage](#usage)
6. [Visualizations & Results](#Visualizations-and-Results)  
7. [Model Performance](#model-performance)  
8. [Real-World Potential & Tangible Benefits](#real-world-potential--tangible-benefits)  
9. [Further Innovations & Expansion Plans](#further-innovations--expansion-plans)  
10. [License](#license)  
11. [Acknowledgments](#acknowledgments)

---

## Overview
This project fine-tunes the **BERT** model for sentiment analysis on the **IMDb** movie review dataset. Text data is tokenized using BERT’s tokenizer and trained on labeled sentiment data to classify reviews as positive or negative. The baseline model achieves 90.8% accuracy on the test set, with a precision of 88.5%, recall of 93.6%, and an F1 score of 0.91, demonstrating strong and balanced classification performance. A subsequent experiment addresses class imbalance using **SMOTE**, resulting in 91% accuracy and improved class balance, with comprehensive evaluation using multiple performance metrics and visual diagnostics.

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

## Project Structure

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

---

## Usage
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

---

## Visualizations and Results

After training and evaluation, you should see output similar to the following:

**Confusion Matrix**
![image](https://github.com/user-attachments/assets/26a4db32-57e4-4743-a97c-9d1e4e88086b)

This image shows the distribution of true positives, false positives, true negatives, and false negatives.

**ROC Curve**
![image](https://github.com/user-attachments/assets/73d20aaa-eff0-4fb2-834d-307b9a394281)

This plot illustrates the trade-off between the true positive rate (TPR) and false positive rate (FPR) across various threshold settings.

**Precision-Recall Curve**
![image](https://github.com/user-attachments/assets/1cded52a-28cd-40f0-a9c6-f93f023701b7)

Highlights how precision and recall vary across different thresholds and is especially useful for imbalanced datasets.

---

## Model Performance

**Test Accuracy (~91%)**  
Represents the proportion of correct predictions on unseen data and reflects the model’s ability to generalize.

### Experimental Results Summary
| Experiment | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|----|
| Baseline BERT (no rebalancing) | 0.9075 | 0.8849 | 0.9356 | 0.9095 |
| Rebalanced Dataset (SMOTE) | 0.91 | 0.87 | 0.91 | 0.89 |


**Confusion Matrix**

- **True Negatives (TN = 1065)**
  - The model correctly predicted the “negative” class when the actual label was negative.

- **False Positives (FP = 112)**
  - The model incorrectly predicted the “positive” class when the actual label was negative.

- **False Negatives (FN = 71)**
  - The model incorrectly predicted the “negative” class when the actual label was positive.

- **True Positives (TP = 752)**
  - The model correctly predicted the “positive” class when the actual label was positive.

**Key Metrics**
**ROC-AUC (~0.97)**
Measures how well the model can distinguish between classes across all thresholds.  
A score closer to 1.0 indicates excellent separation.

**Precision (~0.87)** 
Out of all predicted positives, 87% are truly positive.  
Higher precision means fewer false alarms.

**Recall (~0.91)**
Out of all actual positives, 91% are correctly identified.  
Higher recall means fewer missed positive cases.

**F1-Score (~0.89)**  
The harmonic mean of precision and recall, balancing both measures.  
A higher F1 indicates better overall classification performance.

---

## Real-World Potential & Tangible Benefits
This solution can extend beyond movie reviews to any domain where analyzing large volumes of text-based feedback is crucial—such as **social media monitoring**, **product reviews**, or **brand reputation management**. Automating sentiment analysis enables organizations to:

- **Respond Quickly**: Track shifts in public opinion in near-real time.  
- **Enhance Strategies**: Refine marketing campaigns or product launches based on feedback trends.  
- **Optimize Engagement**: Tailor interactions to better match user sentiment and needs.

---

## Further Innovations & Expansion Plans
- **Explore Diverse Architectures**  
  Experiment with **DistilBERT**, **RoBERTa**, or **GPT** to boost performance or efficiency.  
- **Data Augmentation**  
  Employ text augmentation techniques (e.g., back-translation, synonym replacement) to enhance model robustness, especially for smaller datasets.  
- **Hyperparameter Tuning**  
  Adjust learning rate, batch size, and other parameters to refine accuracy and speed.  
- **Domain Adaptation**  
  Test this pipeline on additional datasets, such as **Twitter data** or **customer service logs**, to validate its versatility in different contexts.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- **Hugging Face** for the `transformers` library  
- **IMDb** for the dataset  
- **Contributors & Community** for continuous support and ideas
