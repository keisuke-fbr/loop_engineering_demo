import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier
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
    # Cabin deck
    df['CabinDeck'] = df['Cabin'].str[0].fillna('U')
    df['CabinDeck'] = LabelEncoder().fit_transform(df['CabinDeck'])
    # Ticket prefix
    df['TicketPrefix'] = df['Ticket'].str.extract(r'^([A-Za-z]+)', expand=False).fillna('NUM')
    df['TicketPrefix'] = LabelEncoder().fit_transform(df['TicketPrefix'])
    return df

train_p = prepare(all_df[all_df['Survived'].notna()].copy())

feats_base = ['Pclass','Sex','AgeBin','FareBin','Embarked','Title','FamilySize','IsAlone']
feats_ext  = feats_base + ['TicketGroupSize']
feats_full = feats_ext + ['CabinDeck','TicketPrefix']

X_base = train_p[feats_base]
X_ext  = train_p[feats_ext]
X_full = train_p[feats_full]
y = train_p['Survived'].astype(int)

best_cv = 0.8339

# I: XGBoost
xgb_clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
scores_i = cross_val_score(xgb_clf, X_base, y, cv=5, scoring='accuracy')
print(f"I - XGBoost (base feats): {scores_i.mean():.4f}")

# J: XGBoost with full features
xgb2 = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
scores_j = cross_val_score(xgb2, X_full, y, cv=5, scoring='accuracy')
print(f"J - XGBoost (full feats+Cabin+Ticket): {scores_j.mean():.4f}")

# K: Stacking RF+SVC+XGB -> LR meta
rf  = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
svc = Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=42)))])
xgb3 = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           use_label_encoder=False, eval_metric='logloss', random_state=42)
meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
stack = StackingClassifier(
    estimators=[('rf', rf), ('svc', svc), ('xgb', xgb3)],
    final_estimator=Pipeline([('s', StandardScaler()), ('m', meta_lr)]),
    cv=5, passthrough=False
)
scores_k = cross_val_score(stack, X_base, y, cv=5, scoring='accuracy')
print(f"K - Stacking RF+SVC+XGB -> LR: {scores_k.mean():.4f}")

best = max(scores_i.mean(), scores_j.mean(), scores_k.mean())
print(f"\nBest this round: {best:.4f} (need > {best_cv:.4f})")
print(f"Improved: {best > best_cv}")
