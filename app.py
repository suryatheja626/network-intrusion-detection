import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("🚨 Network Intrusion Detection System")

uploaded_file = st.file_uploader("Upload Network Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    network_data = pd.read_csv(uploaded_file)

    if st.button("🔍 Detect Intrusion"):

        network_data['byte_ratio'] = network_data['src_bytes'] / (network_data['dst_bytes'] + 1)
        network_data['total_bytes'] = network_data['src_bytes'] + network_data['dst_bytes']
        network_data['bytes_diff'] = network_data['src_bytes'] - network_data['dst_bytes']

        X = network_data.drop(columns=['label'])
        y = network_data['label']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        outlier_fraction = len(network_data[network_data['label']==1]) / float(len(network_data))

        model = IsolationForest(
            n_estimators=200,
            contamination=outlier_fraction,
            random_state=42
        )

        model.fit(X_scaled)

        scores_prediction = model.decision_function(X_scaled)
        y_pred = model.predict(X_scaled)

        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1

        st.success("Detection Completed ✅")

      
        st.write("### Accuracy:", accuracy_score(y, y_pred))
        st.write("### ROC-AUC Score:", roc_auc_score(y, y_pred))

        st.text("Classification Report")
        st.text(classification_report(y, y_pred))

       
        cm = confusion_matrix(y, y_pred)
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title("Confusion Matrix")
        st.pyplot(fig1)

        
        fig2, ax2 = plt.subplots()
        sns.scatterplot(
            x=X_pca[:,0],
            y=X_pca[:,1],
            hue=y_pred,
            palette={0:'blue',1:'red'},
            ax=ax2
        )
        ax2.set_title("PCA Projection")
        st.pyplot(fig2)