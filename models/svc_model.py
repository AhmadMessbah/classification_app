from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def train_model(model, params, x_train, y_train):
    params= {'C':[0.1,0.15,0.2,0.25,1,10],'gamma':['scale', 'auto'],'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}
    grid_search = GridSearchCV(estimator=SVC(), param_grid=params, verbose=1, n_jobs=-1, scoring="accuracy")
    grid_search.fit(x_train, y_train)
    print(grid_search.best_score_)
    print(grid_search.best_params_)
    return grid_search.best_estimator_



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



