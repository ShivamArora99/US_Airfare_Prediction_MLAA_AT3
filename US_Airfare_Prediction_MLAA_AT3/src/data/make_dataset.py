from statistics import mode
import pandas as pd
from statistics import mode


def extract_departure_time(time_obj):
    """Categorizes a time string into more specific time periods."""
    try:
        hour = time_obj.hour

        if 4 <= hour < 6:
            return 'Dawn'
        elif 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 20:
            return 'Evening'
        else:
            return 'Night'
    except (ValueError, TypeError, IndexError):
        return 'Unknown'


from statistics import mode


def clean_cabin_type(cabin_type_str):
    """
    Cleans the cabin type string, finds the mode, and assigns a score.
    """
    if cabin_type_str:
        cabin_types = cabin_type_str.lower().split('|')

        total_score = 0
        for cabin in cabin_types:
            if cabin == 'coach':
                total_score += 1
            elif cabin == 'premium coach':
                total_score += 2
            elif cabin == 'business':
                total_score += 3
            elif cabin == 'first':
                total_score += 4

        mode_cabin = mode(cabin_types)

        return mode_cabin, total_score  # Return both mode and total score
    else:
        return 'Unknown', 0  # Return 'Unknown' and 0 if cabin_type_str is empty


def impute_distance_by_average(df):
    # Calculate average distances for each route
    average_distances = df.groupby(['startingAirport', 'destinationAirport'])[
        'totalTravelDistance'].mean().reset_index()
    # Merge average distances back into the original DataFrame
    df = pd.merge(df, average_distances, on=['startingAirport', 'destinationAirport'], how='left',
                  suffixes=('', '_avg'))
    # Fill missing values with the calculated averages
    df['totalTravelDistance'] = df['totalTravelDistance'].fillna(df['totalTravelDistance_avg'])
    # Drop the temporary average distance column
    df = df.drop(columns=['totalTravelDistance_avg'])
    return df


def clean_data(df):
    """
    Cleans and preprocesses a DataFrame containing flight data.

    This function performs the following cleaning and transformation steps:

    1. Cleans and standardizes airport codes in 'startingAirport' and 'destinationAirport' columns:
        - Removes leading/trailing whitespace.
        - Converts airport codes to uppercase.
        - Removes any non-alphabet characters.

    2. Extracts date components from 'flightDate' column:
        - Extracts year, month, and day into separate columns.

    3. Processes 'segmentsDepartureTimeRaw' column:
        - Extracts the first departure time if multiple times are present (separated by '|').
        - Converts the departure time to datetime objects (UTC).
        - Categorizes the departure time into time-of-day categories (Dawn, Morning, etc.).

    4. Cleans 'segmentsCabinCode' column:
        - Cleans and standardizes cabin codes.
        - Finds the mode of cabin codes if multiple codes are present.

    Args:
        df: The pandas DataFrame containing the flight data.

    Returns:
        The cleaned and preprocessed DataFrame.
    """
    cleaned_df = pd.DataFrame()
    cleaned_df['startingAirport'] = (
        df['startingAirport']
        .str.strip()
        .str.upper()
        .str.replace('[^A-Z]', '', regex=True)
    )
    # cleaned_df['startingAirport'] = cleaned_df['startingAirport'].astype(str).str.split('|').apply(lambda x: x[0])

    cleaned_df['destinationAirport'] = (
        df['destinationAirport']
        .str.strip()
        .str.upper()
        .str.replace('[^A-Z]', '', regex=True)
    )
    # cleaned_df['destinationAirport'] = cleaned_df['destinationAirport'].astype(str).str.split('|').apply(lambda x: x[0])

    cleaned_df['Departure_Year'] = pd.to_datetime(df['flightDate']).dt.year
    cleaned_df['Departure_Month'] = pd.to_datetime(df['flightDate']).dt.month
    cleaned_df['Departure_Day'] = pd.to_datetime(df['flightDate']).dt.day

    df['segmentsDepartureTimeRaw'] = df['segmentsDepartureTimeRaw'].astype(str).str.split('|').apply(lambda x: x[0])
    df['segmentsDepartureTimeRaw'] = pd.to_datetime(df['segmentsDepartureTimeRaw'], utc=True)
    cleaned_df['depatureTimeCategory'] = df['segmentsDepartureTimeRaw'].apply(extract_departure_time)

    cabin_info = df['segmentsCabinCode'].apply(clean_cabin_type)
    cleaned_df['Cabin_Type'] = [cabin[0] for cabin in cabin_info]  # Extract the cabin type (mode)
    cleaned_df['Cabin_Score'] = [cabin[1] for cabin in cabin_info]

    cleaned_df['totalFare'] = df['totalFare']
    cleaned_df['totalTravelDistance'] = df['totalTravelDistance']

    cleaned_df = impute_distance_by_average(cleaned_df)

    return cleaned_df

