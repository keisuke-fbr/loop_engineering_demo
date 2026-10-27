import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('train.csv')

# Title extraction
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
df['Title'] = df['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'})
df['Title'] = LabelEncoder().fit_transform(df['Title'])

# Family features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Deck from Cabin
df['Deck'] = df['Cabin'].str[0].fillna('U')
df['Deck'] = LabelEncoder().fit_transform(df['Deck'])

# Age: median per Title group
df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
df['Age'] = df['Age'].fillna(df['Age'].median())

# Sex & Embarked
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Interaction features
df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3]).astype(int)
df['Pclass_Sex'] = df['Pclass'] * df['Sex']
df['Age_Pclass'] = df['AgeBin'] * df['Pclass']

features = ['Pclass', 'Sex', 'AgeBin', 'FareBin', 'Embarked', 'Title',
            'FamilySize', 'IsAlone', 'Deck', 'Pclass_Sex', 'Age_Pclass']
X = df[features]
y = df['Survived']

model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      random_state=42, eval_metric='logloss', verbosity=0)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
print(f"Scores: {[round(s,4) for s in scores]}")
