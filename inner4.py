import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

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
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3]).astype(int)
    # Non-redundant additions
    df['CabinKnown'] = df['Cabin'].notna().astype(int)
    df['FarePerPerson'] = np.log1p(df['Fare'] / df['FamilySize'])
    return df

train_p = prepare(all_df[all_df['Survived'].notna()].copy())

feats_base = ['Pclass','Sex','AgeBin','FareBin','Embarked','Title','FamilySize','IsAlone']
feats_cabin = feats_base + ['CabinKnown']
feats_fare  = feats_cabin + ['FarePerPerson']

X_base  = train_p[feats_base]
X_cabin = train_p[feats_cabin]
X_fare  = train_p[feats_fare]
y = train_p['Survived'].astype(int)

best_cv = 0.8339

# O: LightGBM (strong regularization)
lgb_o = lgb.LGBMClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    num_leaves=15, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1
)
scores_o = cross_val_score(lgb_o, X_base, y, cv=5, scoring='accuracy')
print(f"O - LightGBM (base feats, regularized): {scores_o.mean():.4f}")

# P: LightGBM with CabinKnown
lgb_p = lgb.LGBMClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    num_leaves=15, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1
)
scores_p = cross_val_score(lgb_p, X_cabin, y, cv=5, scoring='accuracy')
print(f"P - LightGBM + CabinKnown: {scores_p.mean():.4f}")

# Q: RF+SVC+LR + CabinKnown (conservative)
rf  = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
svc = Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=42)))])
lr  = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
q = VotingClassifier([('rf',rf),('svc',svc),('lr',lr)], voting='soft', weights=[2,2,1])
scores_q = cross_val_score(q, X_cabin, y, cv=5, scoring='accuracy')
print(f"Q - RF+SVC+LR + CabinKnown: {scores_q.mean():.4f}")

best = max(scores_o.mean(), scores_p.mean(), scores_q.mean())
print(f"\nBest this round: {best:.4f} (need > {best_cv:.4f})")
print(f"Improved: {best > best_cv}")
