import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('train.csv')

# iter2 で最良だった特徴量セット
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
df['Title'] = df['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'})
df['Title'] = LabelEncoder().fit_transform(df['Title'])

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3]).astype(int)

features = ['Pclass', 'Sex', 'AgeBin', 'FareBin', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
X = df[features]
y = df['Survived']

rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
svc = Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, kernel='rbf', C=1.0, random_state=42))])

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr), ('svc', svc)],
    voting='soft'
)
scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')

print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
print(f"Scores: {[round(s,4) for s in scores]}")
