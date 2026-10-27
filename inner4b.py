import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')
all_df   = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
all_df['TicketGroupSize'] = all_df['Ticket'].map(all_df['Ticket'].value_counts())

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
    # Better Age imputation: by Pclass + Title
    df['Age'] = df.groupby(['Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3]).astype(int)
    df['CabinKnown'] = df['Cabin'].notna().astype(int)
    return df

train_p = prepare(all_df[all_df['Survived'].notna()].copy())

feats_base = ['Pclass','Sex','AgeBin','FareBin','Embarked','Title','FamilySize','IsAlone']
feats_ext  = feats_base + ['CabinKnown']
X_base = train_p[feats_base]
X_ext  = train_p[feats_ext]
y = train_p['Survived'].astype(int)

best_cv = 0.8339

# R: ExtraTrees + SVC + LR (ExtraTrees often more robust)
et  = ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42)
svc = Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=42)))])
lr  = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
r = VotingClassifier([('et',et),('svc',svc),('lr',lr)], voting='soft', weights=[2,2,1])
scores_r = cross_val_score(r, X_base, y, cv=5, scoring='accuracy')
print(f"R - ExtraTrees+SVC+LR: {scores_r.mean():.4f}")

# S: RF+ET+SVC+LR 4 model ensemble
rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
et2 = ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=0)
svc2 = Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=0)))])
lr2  = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=1.0, max_iter=1000, random_state=0))])
s = VotingClassifier([('rf',rf),('et',et2),('svc',svc2),('lr',lr2)], voting='soft', weights=[2,2,2,1])
scores_s = cross_val_score(s, X_base, y, cv=5, scoring='accuracy')
print(f"S - RF+ET+SVC+LR 4-model: {scores_s.mean():.4f}")

# T: RandomizedSearch on RF to find better hyperparams
rf_rs = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [200, 300, 500],
    'max_depth': [4, 5, 6, 7],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 0.5, 0.7]
}
rs = RandomizedSearchCV(rf_rs, param_dist, n_iter=20, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
rs.fit(X_base, y)
print(f"T - RandomizedSearch RF best: {rs.best_score_:.4f} params={rs.best_params_}")

best = max(scores_r.mean(), scores_s.mean(), rs.best_score_)
print(f"\nBest this round: {best:.4f} (need > {best_cv:.4f})")
print(f"Improved: {best > best_cv}")
