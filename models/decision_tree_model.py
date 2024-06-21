from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def decision_tree_model_maker(criterion="gini",
        splitter="best",
        min_samples_split=2,
        min_samples_leaf=1):
    model = DecisionTreeClassifier(criterion=criterion,
                                   splitter=splitter,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf)
    return model



def decision_tree_trainer(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def decision_tree_tester(model, x_test, y_test):
    pred = model.predict(x_test)
    return classification_report(y_test,pred)


# X,y = load_digits(return_X_y=True)
# x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# model = decision_tree_model_maker()
# model = decision_tree_trainer(model, x_train, y_train)
# print(decision_tree_tester(model, x_test, y_test))