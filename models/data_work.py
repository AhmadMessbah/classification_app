from sklearn.model_selection import train_test_split


def data_splitter(X,y,test_size=20):
    test_size /= 100
    print(test_size)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=0)