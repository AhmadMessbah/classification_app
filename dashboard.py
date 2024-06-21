import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from svc_model_ import *

st.title("Machine Learning Dashboard")

dataset_file = st.file_uploader("Upload Dataset")

if dataset_file:
    df = pd.read_excel(dataset_file)

    st.write(df)

    drop = st.selectbox("Drop Fields", df.columns.values)

    df.drop(columns=drop, inplace=True)

    y = st.selectbox("Select Target (y)", df.columns.values)

    X = df.drop(columns= y)
    y = df[y]

    st.write(X)
    st.write(y)

if st.checkbox("Train Test Split"):
    # test_size = int(st.slider("Test Size %",5,50,20))
    x_train, x_test, y_train, y_test = data_splitter(X,y, test_size= 20)

model_name = st.selectbox("Select Model", ["LogisticRegression", "KNeighborsClassifier", "SVC", "DecisionTreeClassifier", "MLPClassifier"])


match model_name:
    case "LogisticRegression":
        st.write("Logistic Regression")
        model = LogisticRegression()

    case "KNeighborsClassifier":
        n_neighbors = st.text_input("n_neighbors",5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

    case "SVC":
        svc_c = st.text_input("C",1)
        svc_kernel = st.text_input("kernel","poly")
        model = svc_model_maker(svc_c, svc_kernel)
        model = svc_trainer(model, x_train, y_train)
        report = svc_tester(model, x_test, y_test)

    case "DecisionTreeClassifier":
        d_tree_splitter = st.text_input("splitter", "best")
        model = DecisionTreeClassifier(splitter=d_tree_splitter)

    case "MLPClassifier":
        mlp_hidden_layers = st.text_input("hidden_layer_sizes", (100,))
        model = MLPClassifier(hidden_layer_sizes=mlp_hidden_layers)


if st.button("Train"):
    st.toast("Wait for training ...")
    info_text = st.write("Wait for training ...")
    model.fit()