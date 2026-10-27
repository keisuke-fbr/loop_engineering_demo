import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import xgboost as xgb

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
    # Better Age imputation: by Pclass + Title group
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
feats_ext  = feats_base + ['CabinKnown','TicketGroupSize']
X_base = train_p[feats_base]
X_ext  = train_p[feats_ext]
y = train_p['Survived'].astype(int)

best_cv = 0.8339

# U: RF(depth4, tuned)+SVC+LR — applying RandomizedSearch best RF params
rf_u = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2, max_features='sqrt', random_state=42)
svc_u = Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=42)))])
lr_u  = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
u = VotingClassifier([('rf',rf_u),('svc',svc_u),('lr',lr_u)], voting='soft', weights=[2,2,1])
scores_u = cross_val_score(u, X_base, y, cv=5, scoring='accuracy')
print(f"U - RF(depth4,tuned)+SVC+LR: {scores_u.mean():.4f}")

# V: XGBoost well-tuned (from XGBoost documentation best practices for small data)
xgb_v = xgb.XGBClassifier(
    n_estimators=1000, max_depth=3, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.7,
    min_child_weight=5, gamma=0.1,
    reg_alpha=0.05, reg_lambda=1.0,
    eval_metric='logloss', random_state=42
)
scores_v = cross_val_score(xgb_v, X_base, y, cv=5, scoring='accuracy')
print(f"V - XGBoost (well-tuned, slow lr): {scores_v.mean():.4f}")

# W: Voting: RF(depth4)+SVC(C=1.5)+LR(C=0.8) with ext features
rf_w = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=42)
svc_w = Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.5, random_state=42)))])
lr_w  = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=0.8, max_iter=1000, random_state=42))])
w = VotingClassifier([('rf',rf_w),('svc',svc_w),('lr',lr_w)], voting='soft', weights=[2,2,1])
scores_w = cross_val_score(w, X_ext, y, cv=5, scoring='accuracy')
print(f"W - RF(d4)+SVC(C=1.5)+LR(C=0.8)+ext: {scores_w.mean():.4f}")

best = max(scores_u.mean(), scores_v.mean(), scores_w.mean())
print(f"\nBest this round: {best:.4f} (need > {best_cv:.4f})")
print(f"Improved: {best > best_cv}")
