from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def data_splitter(X,y,test_size=20):
    test_size /= 100
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=0)

def svc_model_maker(C=1.0, kernel="rbf", degree=3, gamma="scale"):
    model = SVC(C=C,
                kernel=kernel,
                degree=degree,
                gamma=gamma)
    return model


def svc_trainer(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def svc_tester(model, x_test, y_test):
    predict = model.predict(x_test)
    return classification_report(y_test, predict)



