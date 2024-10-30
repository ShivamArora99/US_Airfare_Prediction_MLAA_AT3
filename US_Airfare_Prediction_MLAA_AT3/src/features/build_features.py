from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
import joblib
from scipy.stats.mstats import winsorize
import  pandas as pd

def cyclical_encode(data, col):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max(data[col]))
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max(data[col]))
    return data


def pre_procedss_airport_data(df):
    label_encoder = LabelEncoder()
    combined_airports = pd.concat([df['startingAirport'], df['destinationAirport']])
    label_encoder.fit(combined_airports)
    df['startingAirport_encoded'] = label_encoder.transform(df['startingAirport'])
    df['destinationAirport_encoded'] = label_encoder.transform(df['destinationAirport'])

    with open('airport_enoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    df = cyclical_encode(df, 'Departure_Month')
    df = cyclical_encode(df, 'Departure_Day')
    df = cyclical_encode(df, 'Departure_Year')

    # Create a OneHotEncoder object
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(df[['depatureTimeCategory', 'Cabin_Type']])

    joblib.dump(encoder, 'onehot_encoder.joblib')
    encoded_features = encoder.transform(df[['depatureTimeCategory', 'Cabin_Type']])

    encoded_df = pd.DataFrame(encoded_features,
                              columns=encoder.get_feature_names_out(['depatureTimeCategory', 'Cabin_Type']))

    # Concatenate the one-hot encoded features with the original DataFrame
    df = pd.concat([df, encoded_df], axis=1)

    df['totalFare'] = winsorize(df['totalFare'], limits=[0.05, 0.05])

    return df

# Usage
# Airport_df_pre_processed = pre_procedss_airport_data(Airport_df_cleaned)