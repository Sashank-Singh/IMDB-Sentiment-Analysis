# IMDB Sentiment Analysis

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project is a sentiment analysis model that classifies movie reviews from the IMDB dataset as either **positive** or **negative**. The dataset contains 50,000 movie reviews labeled accordingly. The objective is to apply Natural Language Processing (NLP) techniques to preprocess the text data, vectorize it, and then use machine learning to train a model for sentiment classification.

## Features
- Clean and preprocess textual data (removing HTML tags, punctuation, stop words, etc.).
- Vectorize the text using **TF-IDF**.
- Train a **Logistic Regression** model to classify the sentiment of the reviews.
- Evaluate the model using metrics like **accuracy**, **precision**, **recall**, and **F1-score**.
- Visualize model performance using a **confusion matrix** and **ROC curve**.

## Technologies Used
- **Python**: Core programming language.
- **Jupyter Notebook**: For creating and running the notebook for analysis.
- **Scikit-learn**: For model building and evaluation.
- **Pandas & NumPy**: For data manipulation.
- **Matplotlib & Seaborn**: For data visualization.
- **NLTK (Natural Language Toolkit)**: For text preprocessing.
- **Git**: Version control for tracking changes.
- **GitHub**: Repository for storing and sharing the project.

## Data
The dataset used in this project is the **IMDB Dataset of 50K Movie Reviews**, available on [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

- The dataset consists of two columns:
  - `review`: The text of the movie review.
  - `sentiment`: The label indicating whether the review is positive or negative.

## Installation

To run this project locally, follow these steps:

## Run this in JupyNoteBook for best visualization.


## Or use 

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sashank-Singh/IMDB-Sentiment-Analysis.git
   cd IMDB-Sentiment-Analysis

## Usage

### Running the Notebook:

The Jupyter Notebook is organized into several sections to guide you through the entire process of sentiment analysis:

1. **Data Preprocessing**:
   - Cleaning the dataset by removing HTML tags, special characters, and stop words.
   - Applying lemmatization to reduce words to their base form.
   
2. **Feature Extraction using TF-IDF**:
   - Converting text data into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).

3. **Model Training and Evaluation**:
   - Training a Logistic Regression model on the vectorized data.
   - Evaluating the model using metrics such as accuracy, precision, recall, and F1-score.
   
4. **Visualizations**:
   - Displaying a Confusion Matrix and ROC Curve to understand the model's performance in detail.

### Train and Test the Model:

You can modify the model or add more machine learning algorithms as needed. The flexibility of the notebook allows you to experiment with different classifiers or preprocessing techniques.


