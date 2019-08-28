# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model


def set_missing_ages(df):
    """
    使用RandomForestRegressor拟合缺失的age值
    """
    # print(df.info())
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # print('age_df type: ', type(age_df))

    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    y = known_age[:, 0]
    X = known_age[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    predicted_ages = rfr.predict(unknown_age[:, 1:])
    df.loc[df.Age.isnull(), 'Age'] = predicted_ages

    # print(df.info())

    return df, rfr


def set_Cabin_type(df):
    df.loc[df.Cabin.notnull(), 'Cabin'] = 'Yes'
    df.loc[df.Cabin.isnull(), 'Cabin'] = 'No'

    return df


def features_dummies(df):
    """
    对类目型特征做特征因子化
    """

    dummies_Cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df = pd.concat([df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'Sex', 'Pclass'], axis=1, inplace=True)

    return df


def data_normalize(df):
    """
    特征归一化处理
    """
    scaler = preprocessing.StandardScaler()

    age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)

    fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

    return df, scaler, age_scale_param, fare_scale_param


def data_preprocess(df):
    df, rfr = set_missing_ages(df)
    df = set_Cabin_type(df)
    df = features_dummies(df)
    df, scaler, age_scale_param, fare_scale_param = data_normalize(df)

    df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    # df = df.values

    return df, rfr, scaler, age_scale_param, fare_scale_param


if __name__ == '__main__':
    #load data
    data_train = pd.read_csv('./data/train.csv')
    # print(data_train.describe())
    # print(data_train.Cabin)
    # print(data_train.info())

    train_df, rfr, scaler, age_scale_param, fare_scale_param = data_preprocess(data_train)
    train_np = train_df.values
    y = train_np[:, 0]
    X = train_np[:, 1:]

    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    data_test = pd.read_csv('./data/test.csv')
    test_df = data_test
    #Fare值缺失处理
    test_df.loc[test_df.Fare.isnull(), 'Fare'] = 0
    #Age值缺失处理
    age_df = test_df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    unknown_age = age_df[age_df.Age.isnull()].values
    X = unknown_age[:, 1:]
    predicted_ages = rfr.predict(X)
    test_df.loc[test_df.Age.isnull(), 'Age'] = predicted_ages
    #Cabin值缺失处理
    test_df = set_Cabin_type(data_test)
    #特征因子化处理
    test_df = features_dummies(data_test)
    #归一化
    test_df['Age_scaled'] = scaler.fit_transform(test_df['Age'].values.reshape(-1, 1), age_scale_param)
    test_df['Fare_scaled'] = scaler.fit_transform(test_df['Fare'].values.reshape(-1, 1), fare_scale_param)

    test = test_df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(test)
    result = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': predictions.astype(np.int32)})
    result.to_csv("logistic_regression_preditions.csv", index=False)

    temp = pd.DataFrame({"columns": list(train_df.columns)[1:], "coef": list(clf.coef_.T)})
    print(temp)

    # print(data_train.info())
    # # print(data_train.describe())
    # print(data_test.info())
    # print(data_test.describe())






