#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def load_pima_indian(filename):

    # Read the database
    db = pd.read_csv(filename)

    # some preprocessing
    db['Glucose'] = db['Glucose'].replace(0, np.nan)
    db['BloodPressure'] = db['BloodPressure'].replace(0, np.nan)
    db['SkinThickness'] = db['SkinThickness'].replace(0, np.nan)
    db['Insulin'] = db['Insulin'].replace(0, np.nan)
    db['BMI'] = db['BMI'].replace(0, np.nan)
    db = db.fillna(np.mean(db))
    
    # columns rename
    columns_names = [
        'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin',
        'bmi', 'diabetes_pedigree_function', 'age', 'outcome'
    ]
    db.columns = columns_names
    
    # Create X and Y
    X = db.drop(['outcome'], axis = 1)
    y = db['outcome']

    return X,y

def load_life_expectancy(filename):

    # Read the database
    df = pd.read_csv(filename, sep=',', decimal='.')

    # do some cleaning to the column's names
    def clean_columns(column_name):
        return column_name.strip().lower().replace('  ',' ').replace(' ','_')

    df.columns = [clean_columns(col_name) for col_name in df.columns]

    # rename this column
    df.rename(columns={'thinness_1-19_years':'thinness_10-19_years'}, inplace=True)

    # values imputation
    df = df.fillna(df.mean())

    # Encoding
    df['status'] = df['status'].replace(['Developing', 'Developed'],[0, 1])

    df = pd.concat([df, pd.get_dummies(df['country'], prefix='country', drop_first=True)], axis=1)
    df = df.drop(['country'], axis=1)

    # Drop unnecesary columns
    df = df.drop(columns = ['infant_deaths'])

    # Create X and Y
    X = df.drop(['life_expectancy'], axis = 1)
    y = df['life_expectancy']

    print('Predictors: ', X.shape)
    print('Target: ', y.shape)

    return X, y