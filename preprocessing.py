"""This module has the preprocessing steps for train and test sets, including:
    preproc(): preprocessing the data
    sampler(): SMOTE and random under sampling data
    svmsample(): Alternative SMOTE for SVM and random under sampling data
    """
import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def preproc(df):

    scaler = MinMaxScaler()
    df['Amount'] = np.squeeze(scaler.fit_transform(
        np.array(df['Amount']).reshape(-1, 1)))
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y


# def preproc(df):

#     df['timedelta'] = df['Time'].apply(lambda x: datetime.timedelta(seconds=x))
#     df['fraud'] = df['Class'].astype('category')
#     df.sort_values('Time', inplace=True)
#     df.reset_index(inplace=True, drop=True)
#     df['hour'] = (df['Time']//3600).astype('int')
#     df['hour'] = [hour if hour <= 23 else hour-24 for hour in list(df['hour'])]
#     df['hour'] = df['hour'].astype('category')
#     unkwn_features = ['V{}'.format(n+1) for n in range(0, 28)]

#     df = pd.get_dummies(df, drop_first=True)
#     scaler = MinMaxScaler()
#     df['Amount'] = np.squeeze(scaler.fit_transform(
#         np.array(df['Amount']).reshape(-1, 1)))
#     df.rename(columns={'fraud_1': 'fraud'}, inplace=True)
#     features_lst = ['Amount'] + unkwn_features + \
#         ['hour_{}'.format(i) for i in range(1, 24)]
#     X = df[features_lst]
#     y = df['fraud']

#     return X, y


def sampler(X, y, over_pct=0.1, under_pct=0.2):
    over = BorderlineSMOTE(random_state=42, sampling_strategy=over_pct)
    under = RandomUnderSampler(random_state=42, sampling_strategy=under_pct)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X, y = pipeline.fit_resample(X, y)
    return X, y


def svmsampler(X, y, over_pct=0.1, under_pct=1):
    over = SVMSMOTE(random_state=42, sampling_strategy=over_pct)
    under = RandomUnderSampler(random_state=42, sampling_strategy=under_pct)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X, y = pipeline.fit_resample(X, y)
    return X, y
