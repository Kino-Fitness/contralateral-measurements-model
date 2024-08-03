import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import get_image, str_to_tensor, apply_random_augmentation

def get_data():
    df = pd.read_csv('saved/data/data3.csv')
    df = df.head(46)
    df = df.drop([11, 12, 24, 31])
    df = df.drop(['Index', 'Additional Data', 'Verticies', 'Build', 'Right Image', 'Left Image'], axis=1)
    df.head()

    for index, row in df.iterrows():
        print(index)
        df.at[index, 'Joints'] = str_to_tensor(row['Joints'])  
        df.at[index, 'Body Pose'] = str_to_tensor(row['Body Pose'])
        df.at[index, 'Front Image'] = get_image(row['Front Image'])
        df.at[index, 'Back Image'] = get_image(row['Back Image'])

    gender_map = {'Male': 0, 'Female': 1}
    demographic_map = {"White": 0, "Black": 1, "Asian": 2, "Hispanic": 3}

    df['Gender'] = df['Gender'].map(gender_map)
    df['Demographic'] = df['Demographic'].map(demographic_map)

    for index, row in df.iterrows():
        df.at[index, 'Joints'] = row['Joints'].view(1,-1)

    # Split the dataframe into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    #data agumentation
    augmented_train_df = pd.concat([train_df, train_df], ignore_index=True)
    for i in range(2):
        augmented_train_df = pd.concat([augmented_train_df, augmented_train_df], ignore_index=True)

    for index, row in augmented_train_df.iterrows():
        augmented_train_df.at[index, 'Front Image'] = apply_random_augmentation(row['Front Image']).astype(np.float32)
        augmented_train_df.at[index, 'Back Image'] = apply_random_augmentation(row['Back Image']).astype(np.float32)


    train_df = pd.concat([train_df, augmented_train_df], ignore_index=True)

    train_df.to_pickle('saved/dataframes/train_df.pkl')
    test_df.to_pickle('saved/dataframes/test_df.pkl')
