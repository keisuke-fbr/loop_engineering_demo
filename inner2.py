import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

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
    return df

train_p = prepare(all_df[all_df['Survived'].notna()].copy())
test_p  = prepare(all_df[all_df['Survived'].isna()].copy())

feats_ext = ['Pclass','Sex','AgeBin','FareBin','Embarked','Title','FamilySize','IsAlone','TicketGroupSize']
X_ext, y = train_p[feats_ext], train_p['Survived'].astype(int)

rf  = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
svc = Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=42)))])
lr  = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=1.0, max_iter=1000, random_state=42))])

# G: + TicketGroupSize
g = VotingClassifier([('rf',rf),('svc',svc),('lr',lr)], voting='soft', weights=[2,2,1])
scores_g = cross_val_score(g, X_ext, y, cv=5, scoring='accuracy')
print(f"G - RF+SVC+LR + TicketGroupSize: {scores_g.mean():.4f}")

# H: RF+SVC+LR, SVC C=1.5, RF depth=5 seed変え
rf2 = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=2, random_state=0)
svc2= Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.5, random_state=0)))])
lr2 = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=1.0, max_iter=1000, random_state=0))])
feats_base = ['Pclass','Sex','AgeBin','FareBin','Embarked','Title','FamilySize','IsAlone']
X_base = train_p[feats_base]
h = VotingClassifier([('rf',rf2),('svc',svc2),('lr',lr2)], voting='soft', weights=[2,2,1])
scores_h = cross_val_score(h, X_base, y, cv=5, scoring='accuracy')
print(f"H - RF(n500,seed0)+SVC(C=1.5)+LR: {scores_h.mean():.4f}")

best = max(scores_g.mean(), scores_h.mean())
print(f"\nBest this round: {best:.4f} (need > 0.8339)")
