import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold

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
    df['TitleRaw'] = df['Title'].copy()
    df['Title'] = LabelEncoder().fit_transform(df['Title'])
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Age'] = df.groupby(['Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3]).astype(int)
    df['CabinKnown'] = df['Cabin'].notna().astype(int)
    # LastName for family grouping
    df['LastName'] = df['Name'].str.split(',').str[0].str.strip()
    return df

train_p = prepare(all_df[all_df['Survived'].notna()].copy())
train_p['Survived_int'] = train_p['Survived'].astype(int)

feats_base = ['Pclass','Sex','AgeBin','FareBin','Embarked','Title','FamilySize','IsAlone']
feats_cabin = feats_base + ['CabinKnown']

X_base = train_p[feats_base]
X_cabin = train_p[feats_cabin]
y = train_p['Survived_int']

best_cv = 0.8339

# X: HistGradientBoosting (handles categoricals, missing values natively)
hgb = HistGradientBoostingClassifier(
    max_iter=500, max_depth=4, learning_rate=0.05,
    min_samples_leaf=20, l2_regularization=0.1,
    random_state=42
)
scores_x = cross_val_score(hgb, X_base, y, cv=5, scoring='accuracy')
print(f"X - HistGradientBoosting: {scores_x.mean():.4f}")

# Y: HistGradBoost + CabinKnown
hgb2 = HistGradientBoostingClassifier(
    max_iter=500, max_depth=4, learning_rate=0.05,
    min_samples_leaf=20, l2_regularization=0.1,
    random_state=42
)
scores_y = cross_val_score(hgb2, X_cabin, y, cv=5, scoring='accuracy')
print(f"Y - HistGradBoost + CabinKnown: {scores_y.mean():.4f}")

# Z: Family target encoding (LOO-style) — proper, no leakage
# For each CV fold, compute family survival rate only from training fold
def family_survival_cv(df, y_series, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(df))
    for tr_idx, val_idx in skf.split(df, y_series):
        tr_names = df.iloc[tr_idx]['LastName']
        tr_surv  = y_series.iloc[tr_idx]
        surv_rate = tr_surv.groupby(tr_names.values).mean()
        global_rate = tr_surv.mean()
        val_names = df.iloc[val_idx]['LastName']
        oof[val_idx] = val_names.map(surv_rate).fillna(global_rate).values
    return oof

# Generate family survival rate feature
fam_surv = family_survival_cv(train_p, y)
feats_fam = feats_base + ['FamSurvRate']
X_fam = X_base.copy()
X_fam['FamSurvRate'] = fam_surv

rf_z  = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
svc_z = Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=42)))])
lr_z  = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
z = VotingClassifier([('rf',rf_z),('svc',svc_z),('lr',lr_z)], voting='soft', weights=[2,2,1])
# Note: FamSurvRate here is cross-fold LOO, so no leakage within the CV
scores_z = cross_val_score(z, X_fam, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy')
print(f"Z - RF+SVC+LR + FamilySurvRate(LOO): {scores_z.mean():.4f}")

best = max(scores_x.mean(), scores_y.mean(), scores_z.mean())
print(f"\nBest this round: {best:.4f} (need > {best_cv:.4f})")
print(f"Improved: {best > best_cv}")
