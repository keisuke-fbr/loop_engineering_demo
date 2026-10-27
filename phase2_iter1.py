import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

def prepare(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'})
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

features = ['Pclass', 'Sex', 'AgeBin', 'FareBin', 'Embarked', 'Title', 'FamilySize', 'IsAlone']

train = prepare(pd.read_csv('train.csv'))
test  = prepare(pd.read_csv('test.csv'))
X_train, y_train = train[features], train['Survived']
X_test = test[features]

# GridSearch でハイパーパラメータ最適化
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [5, 6, 7, 8],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=42)
gs = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)

print(f"Best params: {gs.best_params_}")
print(f"CV Accuracy: {gs.best_score_:.4f}")

gs.best_estimator_.fit(X_train, y_train)
preds = gs.best_estimator_.predict(X_test)
pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': preds}).to_csv('submission.csv', index=False)
print("submission.csv 生成完了")
