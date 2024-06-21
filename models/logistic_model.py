from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


def data_splitter(X, y, test_size=20):
    test_size /= 100
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=0)


def logistic_model_maker(C=1.0, solver='lbfgs'):
    model = LogisticRegression(C=C, solver=solver)
    return model


def logistic_trainer(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def logistic_tester(model, x_test, y_test):
    predict = model.predict(x_test)
    return classification_report(y_test, predict)


# data = load_iris()
# X = data.data
# y = data.target
#
# X_train, X_test, y_train, y_test = data_splitter(X, y, test_size=3)
#
# model = logistic_model_maker()
#
# model = logistic_trainer(model, X_train, y_train)
#
# report = logistic_tester(model, X_test, y_test)
# print(report)
