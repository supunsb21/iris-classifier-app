import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Page configuration
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load dataset and split
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (k=5)
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# Session state to control navigation
if "page" not in st.session_state:
    st.session_state.page = "welcome"

def go_to_prediction():
    st.session_state.page = "predict"

# Define image paths
image_dir = "images"
setosa_img = os.path.join(image_dir, "setosa.jpg")
versicolor_img = os.path.join(image_dir, "versicolor.jpg")
virginica_img = os.path.join(image_dir, "virginica.jpg")

# Helper function to load image bytes
def load_image_bytes(path):
    with open(path, "rb") as f:
        return f.read()

if st.session_state.page == "welcome":
    st.title("🌸 Welcome to the Iris Classifier App")
    st.markdown("""
    This application uses a **K-Nearest Neighbors (KNN)** machine learning model to classify
    flowers into one of the three Iris species. Learn about each flower below:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        img_bytes = load_image_bytes(setosa_img)
        st.image(img_bytes, caption="Iris Setosa", use_column_width=True)
        st.markdown("**Setosa** 🌱\n\n- Small and delicate.\n- Short petals and sepals.\n- Typically found in cooler climates.")

    with col2:
        img_bytes = load_image_bytes(versicolor_img)
        st.image(img_bytes, caption="Iris Versicolor", use_column_width=True)
        st.markdown("**Versicolor** 🌼\n\n- Medium-sized flowers.\n- Petals vary in length and color.\n- Found in wet meadows and marshes.")

    with col3:
        img_bytes = load_image_bytes(virginica_img)
        st.image(img_bytes, caption="Iris Virginica", use_column_width=True)
        st.markdown("**Virginica** 🌺\n\n- Largest of the three.\n- Long, wide petals.\n- Often found in the eastern U.S.")

    st.markdown("---")
    st.button("Go to Prediction Page →", on_click=go_to_prediction)

elif st.session_state.page == "predict":
    # Sidebar inputs
    st.sidebar.title("🌿 Input Measurements")
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

    # Predict
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_features)[0]
    proba = model.predict_proba(input_features)[0]
    species = iris.target_names[prediction]

    # Main content
    st.title("🌸 Iris Species Classifier")
    st.subheader("Predicted Species")
    st.success(f"{species.capitalize()}")

    # Show corresponding image
    if species == "setosa":
        img_bytes = load_image_bytes(setosa_img)
        st.image(img_bytes, width=400)
    elif species == "versicolor":
        img_bytes = load_image_bytes(versicolor_img)
        st.image(img_bytes, width=400)
    else:
        img_bytes = load_image_bytes(virginica_img)
        st.image(img_bytes, width=400)

    # Display class probabilities
    st.subheader("Prediction Confidence")
    prob_df = pd.DataFrame({
        'Species': iris.target_names,
        'Probability': proba
    })
    st.bar_chart(prob_df.set_index("Species"))

    # Show model accuracy
    st.markdown("---")
    st.caption(f"Model accuracy on test data: **{accuracy:.2%}** with k = {k}")
    st.caption("Built with Streamlit, scikit-learn & 💻")
