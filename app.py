# import time
# import random

# import streamlit as st
# import streamlit.components.v1 as components
# import pandas as pd

# st.map([{"lat": 40, "lon": 70}])

# progress_bar = st.progress(0)  
# for i in range(100):
#     # time.sleep(0.5)
#     progress_bar.progress(i + 1)

# number = st.number_input("Enter the number")
# st.write("Number is:", number)
# text_field = st.text_input("Enter the text")
# st.write("Text field is:", text_field)

import streamlit as st
from sklearn.tree import DecisionTreeClassifier

from utils import generate_data, gini, entropy
from visualization import plot_decision_boundary, plot_split
from tree_utils import show_tree

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Decision Tree Visualizer", layout="wide")

st.title("🌳 Decision Tree Visualizer (Interactive)")
st.markdown("Understand how Decision Trees split data step-by-step")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

dataset = st.sidebar.selectbox("Dataset", ["simple", "moons", "circles"])
noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.2)

criterion = st.sidebar.selectbox("Impurity", ["gini", "entropy"])

max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
min_samples = st.sidebar.slider("Min Samples Split", 2, 20, 2)

# ---------------- DATA ----------------
X, y = generate_data(dataset, noise)

# ---------------- MODEL ----------------
model = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples,
    criterion=criterion,
    random_state=42
)

model.fit(X, y)

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs([
    "📊 Data & Splits",
    "📉 Impurity",
    "🌳 Tree Structure"
])

# ================= TAB 1 =================
with tab1:
    st.subheader("Decision Boundary")

    st.markdown("""
    👉 The colored regions show how the tree divides space  
    👉 Each region = one decision rule
    """)

    plot_decision_boundary(model, X, y)

    st.subheader("Manual Split Visualization")

    feature = st.selectbox("Feature", [0, 1])
    threshold = st.slider("Threshold", float(X[:, feature].min()), float(X[:, feature].max()))

    plot_split(X, y, feature, threshold)


# ================= TAB 2 =================
with tab2:
    st.subheader("Impurity Measures")

    g = gini(y)
    e = entropy(y)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Gini Impurity", round(g, 4))

    with col2:
        st.metric("Entropy", round(e, 4))

    st.markdown("""
    🎯 Lower impurity = better split  
    Decision Tree tries to reduce impurity at every step
    """)


# ================= TAB 3 =================
with tab3:
    st.subheader("Tree Structure")

    st.markdown("""
    👉 Each node shows a decision  
    👉 Left = True, Right = False  
    👉 Leaves = Final prediction
    """)

    show_tree(model)