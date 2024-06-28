from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from plotly import graph_objs as go


def mlp_model_maker(mlp_hidden_layers=(100,),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=200,
                    verbose=False):
    model = MLPClassifier(
        hidden_layer_sizes=mlp_hidden_layers,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        verbose=verbose)

    return model


def mlp_trainer(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def mlp_tester(model, x_test, y_test):
    predict = model.predict(x_test)
    return classification_report(y_test, predict)


def mlp_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)
    ConfusionMatrixDisplay = confusion_matrix(y_test, y_pred)
    return confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(y_true, y_pred, class_names):
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix = conf_matrix.astype(int)

    layout = {
        "title": "Confusion Matrix",
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"}
    }

    fig = go.Figure(data=go.Heatmap(z=conf_matrix,
                                    x=class_names,
                                    y=class_names,
                                    hoverongaps=False),
                    layout=layout)
    return fig


