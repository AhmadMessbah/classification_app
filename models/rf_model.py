from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report


def data_splitter(X, y, test_size=10):
    return train_test_split(X, y, test_size=test_size)


def rf_model_maker(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=2, criterion='gini'):
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   criterion=criterion)
    return model


def rf_model_trainer(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def rf_model_tester(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return classification_report(y_pred, y_test)

# X,y = load_digits(return_X_y=True)
# x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.15,stratify=y,random_state=0)
# model = rf_model_maker()
# model = rf_model_trainer(model,x_train,y_train)
# print(rf_model_tester(model, x_test, y_test))
