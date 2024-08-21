## Overview
![rick-and-morty-morty-smith](https://github.com/user-attachments/assets/3a780226-bb0c-425a-9316-3872f465b5ad)

Disease Recognizer uses sentence embeddings generated by the `sentence-transformers/all-MiniLM-L6-v2` model to encode patient symptoms into a high-dimensional space. 
Machine learning algorithms, including Logistic Regression and KMeans Clustering, are employed to classify and group symptoms, ultimately predicting the associated disease.

## Features
![CN_Final](https://github.com/user-attachments/assets/055eb59d-6561-4f9e-9797-db6f97202dc2)

- **Symptom Embedding**: Converts text-based symptoms into embeddings using a pre-trained transformer model.
- **Disease Prediction**: Classifies symptoms into disease categories using Logistic Regression.
- **Clustering**: Groups similar symptoms using KMeans Clustering.
- **Data Visualization**: Visualizes the embedded symptom data using t-SNE plots.
- **Interactive Prediction**: Allows for real-time disease prediction based on new symptom inputs.

## Direct run
Go to this url -> https://disease-recogniser-nlp-team-ais.streamlit.app/

## Locall Installation

1. Clone the Disease-Recogniser repository:

```sh
git clone https://github.com/Abhi2april/Disease-Recogniser
```

2. Change to the project directory:

```sh
cd Disease-Recogniser
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```
4. Enter the terminal:
```sh
streamlit run app.py
```
