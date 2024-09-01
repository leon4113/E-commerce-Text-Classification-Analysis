---

# Text Analytics and Sentiment Analysis

## Overview

This project involves text analytics and sentiment analysis using machine learning techniques. The goal is to analyze text data from e-commerce product reviews, classify sentiments, and improve prediction accuracy through various preprocessing methods and machine learning models.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Technologies Used](#technologies-used)
4. [Features](#features)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Methodology](#methodology)
8. [Results](#results)
9. [Contributors](#contributors)

## Introduction

This project aims to utilize text analytics and sentiment analysis techniques to classify e-commerce product reviews into predefined categories. By leveraging natural language processing (NLP) methods, the project focuses on tasks like tokenization, stemming, part-of-speech tagging, and sentiment classification to extract meaningful insights from text data.

## Project Structure

- **Part A: Text Preprocessing and Analysis**
  - Tokenization and stop word removal
  - Word stemming using different algorithms
  - Part-of-speech tagging
  - Sentence probability computations

- **Part B: Machine Learning for Sentiment Classification**
  - Dataset and Exploratory Data Analysis (EDA)
  - Model selection and training (Random Forest, Naive Bayes, Ridge Classifier)
  - Hyperparameter tuning
  - Model evaluation and comparison

## Technologies Used

- **Programming Language**: Python
- **Libraries and Frameworks**:
  - NLTK for natural language processing tasks
  - Scikit-learn for machine learning models
  - Pandas and NumPy for data manipulation
  - Matplotlib and Seaborn for data visualization
- **Jupyter Notebook**: For interactive development and analysis

## Features

- **Text Preprocessing**: Includes tokenization, stop word removal, and stemming.
- **Sentiment Classification**: Using various machine learning models to classify text data into sentiment categories.
- **Model Evaluation**: Comprehensive evaluation using metrics like accuracy, precision, recall, and F1-score.
- **Hyperparameter Tuning**: Optimizing model performance using GridSearchCV.

## Installation

1. **Clone the Repository**: Clone the project repository from GitHub to your local machine.

   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. **Set Up Virtual Environment**: Create and activate a virtual environment.

   ```bash
   python -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   ```

3. **Install Dependencies**: Install the required Python libraries.

   ```bash
   pip install -r requirements.txt
   ```

4. **Run Jupyter Notebook**: Start the Jupyter Notebook server to run the analysis.

   ```bash
   jupyter notebook
   ```

## Usage

1. **Data Preprocessing**: Use the provided Jupyter Notebook to preprocess text data. This includes tokenization, removing stop words, and stemming.
2. **Model Training**: Train the different models (Random Forest, Naive Bayes, Ridge Classifier) using the preprocessed data.
3. **Evaluation**: Evaluate the models using accuracy, precision, recall, and F1-score metrics. Use confusion matrices for detailed error analysis.
4. **Hyperparameter Tuning**: Use the GridSearchCV tool to fine-tune model parameters for better accuracy.

## Methodology

1. **Data Collection**: E-commerce product reviews are collected and organized into a dataset.
2. **Text Preprocessing**: The raw text data is processed using NLP techniques such as tokenization, stop word removal, and stemming.
3. **Model Selection**: Multiple machine learning models are trained, including Random Forest, Naive Bayes, and Ridge Classifier.
4. **Model Training and Evaluation**: Models are trained on the training dataset and evaluated using a separate test dataset. Performance metrics are used to compare models.
5. **Hyperparameter Optimization**: Grid search is applied to find the best hyperparameter settings for the models.

## Results

- **Naive Bayes**: Achieved an accuracy of 95.76%, demonstrating robust performance in text classification tasks.
- **Random Forest**: Achieved an accuracy of 89%, showing strong capabilities in avoiding misclassifications.
- **Ridge Classifier**: Achieved the highest accuracy of 98% after hyperparameter tuning, making it the best performing model in this project.

## Contributors

- **Ariel Amerigo Joe Banua** (TP063209): Text preprocessing, Random Forest model training, evaluation.
- **Leonardo** (TP064705): Sentence probability computations, Ridge Classifier training and evaluation.
- **Nathaniel Sudiono** (TP063926): POS tagging, Naive Bayes model training, hyperparameter tuning.

For any queries or contributions, please contact the project team.

---
