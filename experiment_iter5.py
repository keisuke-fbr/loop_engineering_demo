import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('train.csv')

# Title
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
df['Title'] = df['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'})
df['Title'] = LabelEncoder().fit_transform(df['Title'])

# Family
df['LastName'] = df['Name'].str.split(',').str[0]
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['FamilyGroup'] = df['FamilySize'].apply(lambda x: 'single' if x==1 else ('small' if x<=4 else 'large'))
df['FamilyGroup'] = LabelEncoder().fit_transform(df['FamilyGroup'])

# Family survival rate (train only)
family_survival = df.groupby('LastName')['Survived'].transform('mean')
df['FamilySurvivalRate'] = family_survival

# Ticket prefix
df['TicketPrefix'] = df['Ticket'].str.extract(r'([A-Za-z]+)', expand=False).fillna('NUM')
df['TicketPrefix'] = LabelEncoder().fit_transform(df['TicketPrefix'])

# Name length
df['NameLength'] = df['Name'].str.len()

# Deck
df['Deck'] = df['Cabin'].str[0].fillna('U')
df['Deck'] = LabelEncoder().fit_transform(df['Deck'])

# Sex, Age, Fare, Embarked
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3]).astype(int)

# Interactions
df['Pclass_Sex'] = df['Pclass'] * df['Sex']
df['IsChild'] = (df['Age'] < 12).astype(int)
df['IsWoman'] = ((df['Sex'] == 0) & (df['Age'] >= 18)).astype(int)  # female adult

features = ['Pclass', 'Sex', 'AgeBin', 'FareBin', 'Embarked', 'Title',
            'FamilySize', 'IsAlone', 'FamilyGroup', 'FamilySurvivalRate',
            'TicketPrefix', 'NameLength', 'Deck', 'Pclass_Sex', 'IsChild', 'IsWoman']

X = df[features]
y = df['Survived']

rf = RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=2, random_state=42)
gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                                 subsample=0.8, random_state=42)
lr = Pipeline([('scaler', StandardScaler()),
               ('clf', LogisticRegression(C=0.5, max_iter=1000, random_state=42))])

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
    voting='soft', weights=[2, 2, 1]
)
scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')

print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
print(f"Scores: {[round(s,4) for s in scores]}")
