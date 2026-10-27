import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

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
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3]).astype(int)
    return df

features_base = ['Pclass','Sex','AgeBin','FareBin','Embarked','Title','FamilySize','IsAlone']
train = prepare(pd.read_csv('train.csv'))
test  = prepare(pd.read_csv('test.csv'))
X, y  = train[features_base], train['Survived'].astype(int)

best_cv = 0.8339  # Sub1のベスト

# D: RF+SVC+LR+GB 4モデル, weights調整
rf  = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
gb  = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42)
svc = Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.5, random_state=42)))])
lr  = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=0.5, max_iter=1000, random_state=42))])
d = VotingClassifier([('rf',rf),('gb',gb),('svc',svc),('lr',lr)], voting='soft', weights=[2,2,2,1])
scores_d = cross_val_score(d, X, y, cv=5, scoring='accuracy')
print(f"D - RF+GB+SVC+LR (C=1.5, gb): {scores_d.mean():.4f}")

# E: SVC C=2.0 のみ
svc2 = Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42)))])
scores_e = cross_val_score(svc2, X, y, cv=5, scoring='accuracy')
print(f"E - SVC(C=2.0) only: {scores_e.mean():.4f}")

# F: RF深め + SVC + LR, max_depth=7
rf3 = RandomForestClassifier(n_estimators=400, max_depth=7, min_samples_leaf=1, random_state=42)
f = VotingClassifier([('rf',rf3),('svc',svc),('lr',lr)], voting='soft', weights=[3,2,1])
scores_f = cross_val_score(f, X, y, cv=5, scoring='accuracy')
print(f"F - RF(depth7)+SVC+LR: {scores_f.mean():.4f}")

best = max(scores_d.mean(), scores_e.mean(), scores_f.mean())
print(f"\nBest this round: {best:.4f} (prev best: {best_cv:.4f})")
print(f"Improved: {best > best_cv}")
