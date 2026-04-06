from sklearn.tree import export_graphviz
import streamlit as st

def show_tree(model):
    dot_data = export_graphviz(
        model,
        filled=True,
        feature_names=["X1", "X2"],
        class_names=["Class 0", "Class 1"]
    )
    st.graphviz_chart(dot_data)