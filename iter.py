import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

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

features = ['Pclass','Sex','AgeBin','FareBin','Embarked','Title','FamilySize','IsAlone']
train = prepare(pd.read_csv('train.csv'))
test  = prepare(pd.read_csv('test.csv'))
X, y  = train[features], train['Survived'].astype(int)

mlp = Pipeline([('s', StandardScaler()),
                ('m', MLPClassifier(hidden_layer_sizes=(64,32), activation='relu',
                                    alpha=0.01, max_iter=500, random_state=42))])
rf  = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
svc = Pipeline([('s', StandardScaler()),
                ('m', CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=42)))])

model = VotingClassifier([('mlp',mlp),('rf',rf),('svc',svc)], voting='soft', weights=[1,2,2])
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV: {scores.mean():.4f} (+/- {scores.std():.4f})")

model.fit(X, y)
preds = model.predict(test[features]).astype(int)
sub = pd.DataFrame({'PassengerId': test['PassengerId'].astype(int), 'Survived': preds})
sub.to_csv('submission.csv', index=False)
assert sub.shape[0] == 418
assert sub['Survived'].dtype in ['int32','int64']
print("checklist OK")
