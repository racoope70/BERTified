# Sentiment Analysis with BERT

**Table of Contents**
1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Project Structure](#project-structure)  
5. [Usage](#usage)
6. [Visualizations & Results](#visualizations-and-results)
7. [Model Performance](#model-performance)  
8. [Real-World Potential & Tangible Benefits](#real-world-potential--tangible-benefits)  
9. [Further Innovations & Expansion Plans](#further-innovations--expansion-plans)  
10. [License](#license)  
11. [Acknowledgments](#acknowledgments)

---

## Overview

This project fine-tunes a **BERT (Bidirectional Encoder Representations from Transformers)** model for binary sentiment classification on the **IMDb movie review dataset**. Movie reviews are tokenized using BERT’s tokenizer and trained via supervised learning to predict **positive** or **negative** sentiment.

Two modeling configurations are evaluated:

- A **baseline BERT model** trained on a randomly sampled dataset  
- A **rebalanced BERT model** designed to improve class symmetry and optimize precision–recall trade-offs  

The baseline model achieves **89.7% accuracy**, with strong recall and excellent ranking performance (**ROC–AUC = 0.97**), indicating robust separation between sentiment classes.  
The rebalanced model improves overall accuracy to **91%**, increases precision, and produces a more balanced confusion matrix while maintaining a high ROC–AUC score.

Together, these results highlight the impact of class distribution on classification behavior and demonstrate how targeted rebalancing can improve decision quality without sacrificing model discrimination.

---

## Features
- **Data Preprocessing**  
  - Cleans and tokenizes raw movie reviews for BERT compatibility.  
- **Model Training**  
  - Fine-tunes a pre-trained BERT model on the labeled sentiment data.  
- **Class Imbalance Handling**  
  - Applies dataset rebalancing techniques to mitigate skewed class distributions and improve precision–recall trade-offs.
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
│   ├── imdb_bert_sentiment_baseline.ipynb
│   ├── imdb_bert_sentiment_rebalanced.ipynb
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
  - Open `notebooks/imdb_bert_sentiment_baseline.ipynb` or `imdb_bert_sentiment_rebalanced.ipynb`.
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

To ensure a consistent and transparent comparison, both the **baseline** and **rebalanced** BERT models are evaluated using the same diagnostic plots:

- Confusion Matrix  
- ROC Curve  
- Precision–Recall Curve  

These visualizations provide insight into error distribution, ranking performance, and precision–recall trade-offs beyond aggregate metrics.

---

### Baseline Model (No Rebalancing)

**Confusion Matrix**  
![Baseline Confusion Matrix](images/Baseline%20Confusion%20Matrix.png)

The baseline model demonstrates a recall-oriented behavior, correctly identifying most positive reviews while allowing a higher number of false positives.

**ROC Curve (AUC = 0.97)**  
![Baseline ROC Curve](images/Baseline%20ROC%20Curve.png)

A high ROC–AUC indicates excellent class separability and strong ranking performance across all thresholds.

**Precision–Recall Curve**  
![Baseline Precision-Recall Curve](images/Baseline%20Precision%20Recall%20Curve.png)

The curve shows consistently high recall, confirming the model’s effectiveness at capturing positive sentiment.

---

### Rebalanced Model

**Confusion Matrix**  
![Rebalanced Confusion Matrix](images/Rebalance%20Confusion%20Matrix.png)

After rebalancing, the confusion matrix becomes more symmetric, reducing false positives and improving overall classification balance.

**ROC Curve (AUC ≈ 0.97)**  
![Rebalanced ROC Curve](images/Rebalanced%20ROC%20Curve.png)

The rebalanced model maintains strong discriminative power, with ROC–AUC comparable to the baseline model.

**Precision–Recall Curve**  
![Rebalanced Precision-Recall Curve](images/Rebalance%20Precision%20Recall.png)

Precision improves across higher recall regions, indicating fewer false alarms and a more conservative positive prediction strategy.

### Interpretation

- The **baseline model** prioritizes recall, making it effective for applications where missing positive sentiment is costly.  
- The **rebalanced model** improves precision and overall accuracy, resulting in a more evenly distributed error profile.  

Together, these visualizations highlight the trade-offs between recall and precision and demonstrate the impact of dataset rebalancing on model behavior.


---

## Model Performance

### Baseline Model (No Rebalancing)

- **Accuracy:** 89.7%  
- **Precision:** 87.2%  
- **Recall:** 93.0%  
- **F1 Score:** 0.90  
- **ROC–AUC:** 0.967  

This configuration prioritizes **recall**, making it effective at capturing positive sentiment while minimizing false negatives. The high ROC–AUC indicates strong ranking performance across classification thresholds.

---

### Rebalanced Model

- **Accuracy:** 91.0%  
- **Precision:** 94.0%  
- **Recall:** 88.0%  
- **F1 Score:** 0.91  
- **ROC–AUC:** ~0.97  

The rebalanced model improves overall **accuracy and precision** while maintaining strong recall, resulting in a more **symmetric confusion matrix** and better precision–recall trade-offs.

---

### Experimental Summary

| Experiment        | Accuracy | Precision | Recall | F1  | ROC–AUC |
|------------------|----------|-----------|--------|-----|---------|
| Baseline BERT    | 0.897    | 0.872     | 0.930  | 0.90| 0.97    |
| Rebalanced BERT  | 0.910    | 0.940     | 0.880  | 0.91| ~0.97   |


---

## Real-World Potential & Tangible Benefits

This sentiment analysis pipeline generalizes beyond movie reviews to any domain involving large-scale, unstructured text data, including **product reviews**, **social media streams**, **customer feedback**, and **brand reputation monitoring**.

By automating sentiment classification with a transformer-based model, organizations can:

- **Monitor opinion dynamics at scale**  
  Continuously track sentiment trends across high-volume text sources in near real time.

- **Reduce manual labeling and review costs**  
  Replace labor-intensive qualitative analysis with consistent, reproducible model-driven insights.

- **Support data-driven decision-making**  
  Surface actionable signals for marketing strategy, product feedback loops, customer experience optimization, and risk detection.

The modular design of the pipeline enables straightforward adaptation to new domains, datasets, or downstream analytics workflows, making it suitable for both research experimentation and production-oriented applications.

---
## Further Innovations & Expansion Plans

Potential extensions of this work focus on improving efficiency, robustness, and domain transferability:

- **Model Efficiency & Deployment Trade-offs**  
  Evaluate lighter transformer variants (e.g., **DistilBERT**) to reduce inference latency and memory footprint in production environments.

- **Robustness via Data Augmentation**  
  Apply controlled text augmentation strategies (e.g., back-translation, paraphrasing) to improve generalization under distribution shift.

- **Targeted Hyperparameter Optimization**  
  Systematically tune learning rate schedules, batch sizes, and regularization parameters to balance convergence stability and training efficiency.

- **Domain Transfer & Adaptation**  
  Adapt the pipeline to new text domains (e.g., social media or customer support data) to assess cross-domain performance degradation and retraining requirements.

These extensions are designed to support realistic deployment scenarios and ongoing model lifecycle management rather than isolated benchmark improvements.

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- **Hugging Face** for the `transformers` library  
- **IMDb** for the dataset  
- **Contributors & Community** for continuous support and ideas
