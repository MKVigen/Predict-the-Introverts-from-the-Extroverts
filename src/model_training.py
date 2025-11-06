from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np


def read_data():
    train = pd.read_csv('data/preprocessed/train_df.csv')
    test = pd.read_csv('data/preprocessed/test_df.csv')
    data = pd.read_csv('data/raw/test.csv')

    return train,test,data

def RandomForest(train,test,data):
    X = train.drop(['Personality'], axis=1)
    y = train['Personality']

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100,min_samples_split=2,max_depth=5,random_state=42)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_val)

    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f'accuracy:{acc}')
    print(f'confusion matrix:{cm}')
    print(f'f1 score:{f1}')

    prediction = model.predict(test)
    submission = pd.DataFrame({
        "id": data["id"],
        "Personality": np.where(prediction == 0, "Extrovert", "Introvert")
    })

    submission.to_csv('submissions/RandomForest.csv', index=False)


def LGBMClassifier(train,test,data):
    X = train.drop(['Personality'], axis=1)
    y = train['Personality']

    from lightgbm import LGBMClassifier

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LGBMClassifier(max_depth=5, n_estimators=200, min_samples_split=2, subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbose=-1)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')

    print(f'accuracy:{acc}')
    print(f'confusion matrix:{cm}')
    print(f'f1 score:{f1}')

    prediction = model.predict(test)
    submission = pd.DataFrame({
        "id": data["id"],
        "Personality": np.where(prediction == 0, "Extrovert", "Introvert")
    })

    submission.to_csv('submissions/LGBMClassifier.csv', index=False)

def SVM(train,test,data):
    X = train.drop(['Personality'], axis=1)
    y = train['Personality']

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear',random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f'accuracy:{acc}')
    print(f'confusion matrix:{cm}')
    print(f'f1 score:{f1}')

    prediction = model.predict(test)
    submission = pd.DataFrame({
        "Id": data["id"],
        "Personality": np.where(prediction == 0, "Extrovert", "Introvert")
    })

    submission.to_csv('submissions/SVM.csv', index=False)


def run_model_training():
    train,test,data = read_data()
    print('data reading completed')

    RandomForest(train,test,data)
    print('random forest model prediction completed')

    LGBMClassifier(train,test,data)
    print('lgbmclassifier model prediction completed')

    SVM(train,test,data)
    print('svm model prediction completed')

if __name__ == '__main__':
    run_model_training()

