Sentiment Analysis with BERT
Overview
This project leverages the BERT model to perform sentiment analysis on the IMDb dataset. By processing text data, tokenizing it using BERT's tokenizer, and fine-tuning the model on labeled sentiment data, the project classifies movie reviews as either positive or negative. The analysis demonstrates a significant improvement in performance, achieving a test accuracy of 91%. Updates include handling class imbalance with SMOTE and evaluating model performance using advanced metrics and visualizations.

Features
Data Preprocessing: Efficiently loads and cleans the IMDb dataset, preparing reviews for analysis through tokenization.
Class Imbalance Handling: Utilizes SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic samples for the minority class, improving fairness and performance.
Model Training: Fine-tunes the BERT model for sentiment classification using Hugging Face’s Trainer API.
Evaluation: Incorporates metrics such as confusion matrix, ROC-AUC, and precision-recall curves, along with visualizations to interpret performance.
Interactive Notebooks:
IMDB_Sentiment_Analysis_with_BERT.ipynb: Original notebook for the initial analysis and training.
Results_Rebalance_Dataset.ipynb: Enhanced notebook version that applies SMOTE to address class imbalance and includes advanced evaluation metrics and visualizations.
Memory Optimization: Implements strategies to handle large datasets efficiently without overwhelming resources.

Requirements
Python Version: 3.7+
Libraries:
transformers
pandas
scikit-learn
torch
tensorflow
IMDb Dataset: Ensure the dataset is downloaded and structured appropriately (e.g., aclImdb/train and aclImdb/test).

1) Installation
Clone the Repository:
git clone https://github.com/yourusername/sentiment-analysis-bert.git

2) Navigate to Project Directory:
cd sentiment-analysis-bert

3) Install Dependencies:
pip install -r requirements.txt

Directory Structure
Sentiment-Analysis-BERT/
|
├── data/                    # Folder for the dataset (if provided)
├── src/                     # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
|
├── notebooks/               # Jupyter Notebooks
│   ├── IMDB_Sentiment_Analysis_with_BERT.ipynb
|
├── requirements.txt         # Dependencies
├── README.md                # Project overview

Usage
Option 1: Use the Jupyter Notebook
Open notebooks/IMDB_Sentiment_Analysis_with_BERT.ipynb to run the project interactively.
This notebook covers all steps, from loading the data to evaluating the model.

Option 2: Use the Command Line Scripts
Prepare the Data:

Ensure the IMDb dataset is stored in the data/ directory in the expected format.
Run Preprocessing:

Execute the data_preprocessing.py script to clean and tokenize the dataset.
bash
Copy code
python src/data_preprocessing.py
Train the Model:

Fine-tune the BERT model by running the model_training.py script.
bash
Copy code
python src/model_training.py
Evaluate Performance:

Test the model and view its accuracy using the evaluation.py script.
bash
Copy code
python src/evaluation.py

Model Performance
Training Accuracy: ~93%
Test Accuracy: ~91%
Metrics:
Confusion Matrix: Detailed breakdown of true/false positives and negatives.
ROC-AUC: Strong discrimination with an AUC score of 0.95.
Precision-Recall Curve: Highlights the trade-off between precision and recall, especially for imbalanced datasets.

Contributing
Contributions are welcome! Fork the repository, make your changes, and submit a pull request. All contributions to improving the project are appreciated.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
Hugging Face: For the transformers library.
IMDb: For providing the dataset.
