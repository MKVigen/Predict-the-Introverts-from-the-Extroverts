import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def data_overview(df):
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDescriptive Stats:\n", df.describe())

def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing>0].sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing.index, y=missing.values)
    plt.xticks(rotation=45)
    plt.title("Missing Values per Column")
    plt.show()


def feature_distribution(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, kde = True,x="variable", y="value")
    plt.title("Feature Distribution")
    plt.show()

def sum_func(df):
    result = []
    for _, row in df.iterrows():
        total = 0
        for value in row:
            if pd.isnull(value):
                continue
            elif isinstance(value, (int, float)):
                total += value
            elif isinstance(value, str):
                if value == 'Yes':
                    total += 1
                elif value == 'No':
                    total += 0
        result.append(total)
    return result

def fill_missing_values(df):
    min_sum = df['sum'].min()
    max_sum = df['sum'].max()
    val = (df['sum'] - min_sum) // (max_sum - min_sum)

    if 'Friends_circle_size' in df.columns:
        df['Friends_circle_size'].fillna(df['Friends_circle_size'].mean(),inplace=True)
    if 'Time_spent_Alone' in df.columns:
        df['Time_spent_Alone'] = df['Time_spent_Alone'].fillna(10 - val)
    if 'Stage_fear' in df.columns:
        df['Stage_fear'] = np.where(df['sum'] < 20, 'Yes', 'No')
    if 'Going_outside' in df.columns:
        df['Going_outside'] = df['Going_outside'].fillna(val*10)
    if 'Drained_after_socializing' in df.columns:
        df['Drained_after_socializing'] = np.where(df['sum'] < 20, 'Yes', 'No')
    if 'Social_event_attendance' in df.columns:
        df['Social_event_attendance'] = df['Social_event_attendance'].fillna(val*10)
    if 'Post_frequency' in df.columns:
        df['Post_frequency'] = df['Post_frequency'].fillna(val*10)
    else:
        print('oops')

    return df

def from_float_to_int(df):
    df['Post_frequency'] = df['Post_frequency'].astype(int)
    df['Time_spent_Alone'] = df['Time_spent_Alone'].astype(int)
    df['Social_event_attendance'] = df['Social_event_attendance'].astype(int)
    df['Going_outside'] = df['Going_outside'].astype(int)
    df['sum'] = df['sum'].astype(int)

    return df

def encoding(df):
    label_encoder = LabelEncoder()
    df['Stage_fear'] = label_encoder.fit_transform(df['Stage_fear'])
    df['Drained_after_socializing'] = label_encoder.fit_transform(df['Drained_after_socializing'])

    if 'Personality' in df.columns:
        df['Personality'] = label_encoder.fit_transform(df['Personality'])
    else:
        print('its test data')

    return df

def save_data(train, test):

    import os
    output_dir = '/Users/vigenmkrtchyan/Documents/Introverts from Extroverts/data/preprocessed'
    os.makedirs(output_dir, exist_ok=True)

    # Save files with correct filenames
    train.to_csv(os.path.join(output_dir, 'train_df.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test_df.csv'), index=False)


def run_preprocessing():
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')

    print('overview of train data')
    data_overview(train_df)
    print('overview of test data')
    data_overview(test_df)

    print('plot missing values of train data')
    plot_missing_values(train_df)
    print('plot missing values of test data')
    plot_missing_values(test_df)

    print('some important changes on datasets')
    train_df['sum'] = sum_func(train_df)
    test_df['sum'] = sum_func(test_df)

    print('filling missing values of train data')
    train_df = fill_missing_values(train_df)
    print('filling missing values of test data')
    test_df = fill_missing_values(test_df)

    print('make float dtypes int so as to save memory')
    train_df = from_float_to_int(train_df)
    test_df = from_float_to_int(test_df)

    print('encoding data')
    train_df = encoding(train_df)
    test_df = encoding(test_df)

    print("Data preprocessing complete and files saved to 'data/processed/'.")
    save_data(train_df, test_df)

if __name__ == '__main__':
    run_preprocessing()