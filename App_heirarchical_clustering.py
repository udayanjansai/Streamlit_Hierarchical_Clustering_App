import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(layout="wide")

st.title("🟣 News Topic Discovery Dashboard")
st.write("Hierarchical clustering for automatic news grouping.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

max_features = st.sidebar.slider("TF-IDF Features", 100, 2000, 1000)
svd_dim = st.sidebar.slider("SVD Components", 10, 100, 50)

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

dendro_samples = st.sidebar.slider(
    "Articles for Dendrogram",
    20, 200, 80
)

clusters = st.sidebar.slider("Number of Clusters", 2, 10, 2)

# ---------------- LOAD DATA ----------------
if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except:
        df = pd.read_csv(uploaded_file, encoding="latin-1")

    text_column = df.select_dtypes(include="object").columns[0]
    texts = df[text_column].astype(str)

    st.write(f"Detected text column: **{text_column}**")

    # ---------------- TF-IDF ----------------
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)

    # ---------------- SVD (FIX) ----------------
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    X_reduced = svd.fit_transform(X)

    # ---------------- DENDROGRAM ----------------
    if st.button("🟦 Generate Dendrogram"):
        st.subheader("Dendrogram")

        subset = X_reduced[:dendro_samples]
        Z = linkage(subset, method=linkage_method)

        fig, ax = plt.subplots(figsize=(10,5))
        dendrogram(Z)
        plt.xlabel("Articles")
        plt.ylabel("Distance")
        st.pyplot(fig)

    # ---------------- APPLY CLUSTERING ----------------
    if st.button("🟩 Apply Clustering"):

        model = AgglomerativeClustering(
            n_clusters=clusters,
            linkage=linkage_method
        )

        labels = model.fit_predict(X_reduced)
        df["cluster"] = labels

        # ---------------- PCA VISUALIZATION ----------------
        st.subheader("Cluster Visualization")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_reduced)

        fig, ax = plt.subplots()
        ax.scatter(X_pca[:,0], X_pca[:,1], c=labels)
        st.pyplot(fig)

        # ---------------- CLUSTER SUMMARY ----------------
        st.subheader("Cluster Summary")

        feature_names = vectorizer.get_feature_names_out()
        X_dense = X.toarray()

        summary = []

        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            cluster_vectors = X_dense[idx]
            mean_tfidf = cluster_vectors.mean(axis=0)

            top_indices = np.argsort(mean_tfidf)[-10:]
            keywords = [feature_names[i] for i in top_indices]

            summary.append([
                c,
                len(idx),
                ", ".join(keywords),
                texts.iloc[idx[0]][:120]
            ])

        summary_df = pd.DataFrame(
            summary,
            columns=[
                "Cluster ID",
                "Articles",
                "Top Keywords",
                "Sample Article"
            ]
        )

        st.dataframe(summary_df)

        # ---------------- SILHOUETTE ----------------
        st.subheader("Validation")

        score = silhouette_score(X_reduced, labels)
        st.write(f"Silhouette Score: **{round(score,3)}**")

        st.info(
            "Articles grouped in the same cluster share similar vocabulary "
            "and themes. These clusters can be used for automatic tagging "
            "and recommendations."
        )
