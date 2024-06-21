import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

st.title("KNN Model")

# File uploader for CSV dataset
dataset_file = st.file_uploader("Upload Dataset (CSV)", type="csv")

if dataset_file:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset_file)
    st.write(df)

    # Slider for selecting the test set percentage
    test_size = st.slider("Select Test Set Percentage", min_value=5, max_value=25, value=20, step=1) / 100.0

    # Select target variable
    target = st.selectbox("Select Target (y)", df.columns.values)

    # Features and target
    X = df.drop(columns=target)
    y = df[target]

    # Train-test split
    if st.checkbox("Train Test Split"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.write("Train Set")
        st.write(X_train)
        st.write(y_train)
        st.write("Test Set")
        st.write(X_test)
        st.write(y_test)

        # KNN model
        n_neighbors = st.number_input("Number of Neighbors (k)", min_value=1, max_value=20, value=5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

        if st.button("Train"):
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        # GridSearchCV for KNN
        if st.button("Grid Search for Best K"):
            param_grid = {'n_neighbors': range(1, 21)}
            grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            best_k = grid_search.best_params_['n_neighbors']
            best_score = grid_search.best_score_
            st.write(f"Best Number of Neighbors: {best_k}")
            st.write(f"Cross-validated Accuracy: {best_score * 100:.2f}%")
