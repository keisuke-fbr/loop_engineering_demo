import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def prepare(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['Title'] = LabelEncoder().fit_transform(df['Title'])
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3]).astype(int)
    df['Deck'] = df['Cabin'].str[0].fillna('U')
    df['Deck'] = LabelEncoder().fit_transform(df['Deck'])
    return df

features = ['Pclass','Sex','AgeBin','FareBin','FarePerPerson','Embarked',
            'Title','FamilySize','IsAlone','Deck']

train = prepare(pd.read_csv('train.csv'))
test  = prepare(pd.read_csv('test.csv'))
X_train, y_train = train[features], train['Survived']

lgb = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
rf  = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
lr  = Pipeline([('s', StandardScaler()),
                ('m', LogisticRegression(C=1.0, max_iter=1000, random_state=42))])

ensemble = VotingClassifier([('lgb', lgb), ('rf', rf), ('lr', lr)], voting='soft', weights=[2,2,1])
scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV: {scores.mean():.4f} (+/- {scores.std():.4f})")

ensemble.fit(X_train, y_train)
preds = ensemble.predict(test[features])
pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': preds}).to_csv('submission.csv', index=False)
print("submission.csv 生成完了")
