import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

def prepare(df):
    df = df.copy()
    df['Sex'] = (df['Sex'] == 'female').astype(int)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['Title'] = LabelEncoder().fit_transform(df['Title'])
    return df

features = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','Title']

train = prepare(pd.read_csv('train.csv'))
test  = prepare(pd.read_csv('test.csv'))
X_train, y_train = train[features], train['Survived']

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=0.5, max_iter=1000, random_state=42))
])
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV: {scores.mean():.4f} (+/- {scores.std():.4f})")

model.fit(X_train, y_train)
preds = model.predict(test[features])
pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': preds}).to_csv('submission.csv', index=False)
print("submission.csv 生成完了")
