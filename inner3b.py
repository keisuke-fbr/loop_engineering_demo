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
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3]).astype(int)
    # Domain-specific features
    df['IsChild'] = (df['Age'] < 16).astype(int)
    df['IsWoman'] = (df['Sex'] == 0).astype(int)  # 0=female after LabelEncoder
    df['WomanOrChild'] = ((df['Age'] < 16) | (df['Sex'] == 0)).astype(int)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['FarePerBin'] = pd.qcut(df['FarePerPerson'], q=4, labels=[0,1,2,3], duplicates='drop').astype(int)
    df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex'].astype(str)
    df['Pclass_Sex'] = LabelEncoder().fit_transform(df['Pclass_Sex'])
    return df

train_p = prepare(all_df[all_df['Survived'].notna()].copy())

feats_base = ['Pclass','Sex','AgeBin','FareBin','Embarked','Title','FamilySize','IsAlone']
feats_domain = feats_base + ['IsChild','WomanOrChild','FarePerBin','Pclass_Sex']

X_base = train_p[feats_base]
X_dom  = train_p[feats_domain]
y = train_p['Survived'].astype(int)

best_cv = 0.8339

# L: RF+SVC+LR with domain features
rf  = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
svc = Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=42)))])
lr  = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
l = VotingClassifier([('rf',rf),('svc',svc),('lr',lr)], voting='soft', weights=[2,2,1])
scores_l = cross_val_score(l, X_dom, y, cv=5, scoring='accuracy')
print(f"L - RF+SVC+LR + domain feats: {scores_l.mean():.4f}")

# M: XGB with domain features
xgb_m = xgb.XGBClassifier(
    n_estimators=500, max_depth=3, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    eval_metric='logloss', random_state=42
)
scores_m = cross_val_score(xgb_m, X_dom, y, cv=5, scoring='accuracy')
print(f"M - XGB (domain feats, tuned): {scores_m.mean():.4f}")

# N: Voting RF+SVC+LR+XGB with domain features
rf2 = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
svc2 = Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=42)))])
lr2  = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
xgb_n = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric='logloss', random_state=42)
n = VotingClassifier([('rf',rf2),('svc',svc2),('lr',lr2),('xgb',xgb_n)], voting='soft', weights=[2,2,1,2])
scores_n = cross_val_score(n, X_dom, y, cv=5, scoring='accuracy')
print(f"N - RF+SVC+LR+XGB + domain feats: {scores_n.mean():.4f}")

best = max(scores_l.mean(), scores_m.mean(), scores_n.mean())
print(f"\nBest this round: {best:.4f} (need > {best_cv:.4f})")
print(f"Improved: {best > best_cv}")
