import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------- DECISION BOUNDARY ----------------
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()

    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")

    ax.set_title("Decision Regions (How tree splits space)")
    st.pyplot(fig)


# ---------------- SIMPLE SPLIT VISUAL ----------------
def plot_split(X, y, feature=0, threshold=0):
    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")

    if feature == 0:
        ax.axvline(threshold, color="red", linestyle="--")
    else:
        ax.axhline(threshold, color="red", linestyle="--")

    ax.set_title(f"Split on Feature {feature} at {threshold:.2f}")
    st.pyplot(fig)