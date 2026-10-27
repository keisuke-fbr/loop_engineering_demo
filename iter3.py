import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

def prepare(df):
    df = df.copy()
    df['Sex'] = (df['Sex'] == 'female').astype(int)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

    # Titanic の核心パターンを特徴量化
    df['IsWoman'] = df['Sex']
    df['IsChild'] = (df['Age'] < 15).astype(int)
    df['WomanOrChild'] = ((df['IsWoman'] == 1) | (df['IsChild'] == 1)).astype(int)
    df['HighClass'] = (df['Pclass'] <= 2).astype(int)
    df['WomanHighClass'] = df['IsWoman'] * df['HighClass']
    df['ChildHighClass'] = df['IsChild'] * df['HighClass']

    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['Title'] = LabelEncoder().fit_transform(df['Title'])

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df

features = ['Pclass','Sex','Age','Fare','Embarked','Title',
            'FamilySize','IsAlone','WomanOrChild','WomanHighClass','ChildHighClass']

train = prepare(pd.read_csv('train.csv'))
test  = prepare(pd.read_csv('test.csv'))
X_train, y_train = train[features], train['Survived']

model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                    subsample=0.8, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV: {scores.mean():.4f} (+/- {scores.std():.4f})")

model.fit(X_train, y_train)
preds = model.predict(test[features])
pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': preds}).to_csv('submission.csv', index=False)
print("submission.csv 生成完了")
