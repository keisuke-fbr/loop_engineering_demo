import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')
all_df   = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

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
    df['Age'] = df.groupby(['Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3]).astype(int)
    df['LastName'] = df['Name'].str.split(',').str[0].str.strip()
    return df

train_p = prepare(train_df.copy())
test_p  = prepare(test_df.copy())

feats = ['Pclass','Sex','AgeBin','FareBin','Embarked','Title','FamilySize','IsAlone','FamSurvRate']
y = train_p['Survived'].astype(int)

# Compute LOO family survival rate for training
global_rate = y.mean()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fam_surv_train = np.zeros(len(train_p))
for tr_idx, val_idx in skf.split(train_p, y):
    tr_names = train_p.iloc[tr_idx]['LastName']
    tr_y = y.iloc[tr_idx]
    surv_rate = tr_y.groupby(tr_names.values).mean()
    val_names = train_p.iloc[val_idx]['LastName']
    fam_surv_train[val_idx] = val_names.map(surv_rate).fillna(global_rate).values

# Full family survival rate from all training data for test lookup
full_surv_rate = y.groupby(train_p['LastName'].values).mean()
fam_surv_test = test_p['LastName'].map(full_surv_rate).fillna(global_rate).values

X_train = train_p[feats[:-1]].copy()
X_train['FamSurvRate'] = fam_surv_train
X_test  = test_p[feats[:-1]].copy()
X_test['FamSurvRate']  = fam_surv_test

rf  = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
svc = Pipeline([('s', StandardScaler()), ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=42)))])
lr  = Pipeline([('s', StandardScaler()), ('m', LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
model = VotingClassifier([('rf',rf),('svc',svc),('lr',lr)], voting='soft', weights=[2,2,1])
model.fit(X_train, y)

preds = model.predict(X_test).astype(int)
sub = pd.DataFrame({'PassengerId': test_p['PassengerId'].astype(int), 'Survived': preds})
sub.to_csv('submission.csv', index=False)

assert sub.shape[0] == 418, f"行数エラー: {sub.shape[0]}"
assert sub['Survived'].isin([0,1]).all(), "Survived に 0/1 以外の値あり"
assert sub['Survived'].dtype in ['int32','int64'], "Survived が int でない"
print("チェックリスト OK")
print(sub['Survived'].value_counts().to_dict())
