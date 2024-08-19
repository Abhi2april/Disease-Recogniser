import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

@st.cache
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/mistralai/cookbook/main/data/Symptom2Disease.csv", index_col=0)
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    return df, label_encoder

df, label_encoder = load_data()

# Prepare embeddings
df['embeddings'] = df['text'].apply(lambda x: get_text_embedding(x))
X = df['embeddings'].tolist()
y = df['label_encoded']

# Train the Logistic Regression model
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
clf = LogisticRegression(random_state=0, C=1.0, max_iter=500).fit(train_x, train_y)

# Clustering with KMeans
kmeans_model = KMeans(n_clusters=24, max_iter=1000)
kmeans_model.fit(df['embeddings'].to_list())
df["cluster"] = kmeans_model.labels_

# Streamlit app layout
st.title("Disease Classification and Clustering")

# User input
text = st.text_input("Enter symptoms:", "")
if text:
    embedding = get_text_embedding(text)
    prediction = clf.predict([embedding]).item()
    predicted_disease = label_encoder.inverse_transform([prediction])[0]
    st.write(f"Predicted Disease: {predicted_disease}")

    # Display cluster information
    cluster_label = kmeans_model.predict([embedding])[0]
    st.write(f"Cluster: {cluster_label}")
    st.write("Sample texts in this cluster:")
    sample_texts = df[df.cluster == cluster_label].text.head(3).tolist()
    for t in sample_texts:
        st.write(f"- {t}")

    # Visualize embeddings with t-SNE
    tsne = TSNE(n_components=2, random_state=0).fit_transform(np.array(df['embeddings'].to_list()))
    fig, ax = plt.subplots()
    scatter = ax.scatter(tsne[:, 0], tsne[:, 1], c=df['label_encoded'], cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    st.pyplot(fig)
