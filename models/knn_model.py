import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def start():
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

            # Become commented
            st.write("Train Set")
            st.write(X_train)
            st.write(y_train)
            st.write("Test Set")
            st.write(X_test)
            st.write(y_test)

            # Number of neighbors input
            n_neighbors = st.selectbox("Select how to determine ", ["Input by user", "Grid Search"])


            match n_neighbors:
                case "Input by user":
                    n_neighbors = st.text_input("n_neighbors",5)
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)
                    if st.button("Train"):
                        model.fit(X_train, y_train)
                        accuracy = model.score(X_test, y_test)
                        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

                    

                        # Plot decision boundary if the data has 2 features
                        if X.shape[1] == 2:
                            st.write("Decision Boundary:")
                            x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
                            y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
                            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                                np.arange(y_min, y_max, 0.1))

                            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                            Z = Z.reshape(xx.shape)
                            plt.contourf(xx, yy, Z, alpha=0.4)
                            plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, marker='o', edgecolor='k', s=20)
                            plt.xlabel(X.columns[0])
                            plt.ylabel(X.columns[1])
                            st.pyplot()
                case "Grid Search":
                    
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)


            # GridSearchCV for KNN
            if st.button("Grid Search for Best K"):
                param_grid = {'n_neighbors': range(1, len(X_train) + 1)}
                grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
                grid_search.fit(X_train, y_train)
                best_k = grid_search.best_params_['n_neighbors']
                best_score = grid_search.best_score_
                st.write(f"Best Number of Neighbors: {best_k}")
                st.write(f"Cross-validated Accuracy: {best_score * 100:.2f}%")

                # Train the model with the best k
                best_model = KNeighborsClassifier(n_neighbors=best_k)
                best_model.fit(X_train, y_train)
                Y_predict = best_model.predict_proba(y_test)

                # Plot decision boundary if the data has 2 features
                if X.shape[1] == 2:
                    st.write("Decision Boundary:")
                    x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
                    y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                        np.arange(y_min, y_max, 0.1))

                    Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    plt.contourf(xx, yy, Z, alpha=0.4)
                    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, marker='o', edgecolor='k', s=20)
                    plt.xlabel(X.columns[0])
                    plt.ylabel(X.columns[1])
                    st.pyplot()
