import os
import pickle

import streamlit as st
import pandas as pd
from models import *

st.title("Machine Learning Dashboard")

dataset_file = st.file_uploader("Upload Dataset")

if dataset_file:
    # todo : select file type
    df = pd.read_csv(dataset_file)

    st.write(df)

    drop = st.selectbox("Drop Fields", df.columns.values)

    df.drop(columns=drop, inplace=True)

    y = st.selectbox("Select Target (y)", df.columns.values)

    X = df.drop(columns=y)
    y = df[y]

    st.write(X)
    st.write(y)

if st.checkbox("Train Test Split"):
    test_size = int(st.slider("Test Size %",5,50,20))
    x_train, x_test, y_train, y_test = data_splitter(X, y, test_size=test_size)

model_name = st.selectbox("Select Model",
                          ["LogisticRegression", "KNeighborsClassifier", "SVC", "DecisionTreeClassifier",
                           "MLPClassifier"])

match model_name:
    case "LogisticRegression":
        st.write("Logistic Regression")
        # model = LogisticRegression()

    case "KNeighborsClassifier":
        n_neighbors = st.text_input("n_neighbors", 5)
        # model = KNeighborsClassifier(n_neighbors=n_neighbors)

    case "SVC":
        C = st.selectbox("C", [0.1,0.15,0.2,0.25,1,10], 4)
        solver = st.selectbox("solver", ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 2)
        gamma = st.selectbox("gamma", [0.1,0.15,0.2,0], 0)
        if solver == "Poly":
            degree = st.selectbox("degree", [1, 2, 3, 4, 5, 6, 7, 8, 9],2)
        else:
            degree = 3
        model = svc_model_maker(C, solver,gamma,degree)

    case "DecisionTreeClassifier":
        d_tree_splitter = st.text_input("splitter", "best")
        # model = DecisionTreeClassifier(splitter=d_tree_splitter)

    case "MLPClassifier":
        mlp_hidden_layers = eval(st.text_input("hidden_layer_sizes", (100,)))
        activation = st.selectbox("activation", ["identity", "logistic", "relu", "tanh"], 2)
        solver = st.selectbox("solver", ["lbfgs", "sgd", "adam"], 2)
        model = mlp_model_maker(mlp_hidden_layers, activation, solver)

if st.button("Train"):
    st.toast("Wait for training ...")

    model.fit(x_train, y_train)

    st.toast("Done")

    if st.button("Save Model"):
        print(os.getcwd())
        with open("model.pkl", "wb") as file:
            pickle.dump(model, file)
