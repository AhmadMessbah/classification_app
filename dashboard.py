import os
import pickle
import streamlit as st
import pandas as pd
from models import *

st.title("Machine Learning Dashboard")


style = "<style> .appview-container .main .block-container{ max-width: 100%; padding: 50px;}</style>"""
st.markdown(body=style, unsafe_allow_html=True)

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

    col1, col2 = st.columns([0.9,0.1])

    with col1:
        st.write(X)
    with col2:
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
        svc_c = st.text_input("C", 1)
        svc_kernel = st.text_input("kernel", "poly")
        model = svc_model_maker(svc_c, svc_kernel)
        model = svc_trainer(model, x_train, y_train)
        report = svc_tester(model, x_test, y_test)

    case "DecisionTreeClassifier":
        d_tree_splitter = st.text_input("splitter", "best")
        # model = DecisionTreeClassifier(splitter=d_tree_splitter)

    case "MLPClassifier":
        mlp_hidden_layers = eval(st.text_input("hidden_layer_sizes", (100,)))
        activation = st.selectbox("activation", ["identity", "logistic", "relu", "tanh"], 2)
        solver = st.selectbox("solver", ["lbfgs", "sgd", "adam"], 2)
        verbose=st.selectbox("verbose", [False, True],0)
        print(verbose)
        max_iter = int(st.text_input("max_iter", 200))
        print(max_iter)
        learning_rate_init=float(st.text_input("learning_rate_init", 0.001))
        print(learning_rate_init)
        print(type(learning_rate_init))
        model = mlp_model_maker(mlp_hidden_layers, activation, solver,learning_rate_init, max_iter, verbose)


if st.button("Train"):
    st.toast("Wait for training ...")

    model.fit(x_train, y_train)

    st.toast("Done")
if st.button("Save Model"):
    print(os.getcwd())
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

if st.button("Test"):
        st.toast("Wait for testing ...")

        mlp_trainer(model, x_train, y_train)
        st.write(mlp_tester(model, x_test, y_test))
        st.toast("Done")


