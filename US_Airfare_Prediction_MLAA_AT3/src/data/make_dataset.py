from statistics import mode
import pandas as pd

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


def clean_cabin_type(cabin_type_str):
    """
    Cleans the cabin type string, finds the mode, and assigns a score.
    """
    if cabin_type_str:
        cabin_types = cabin_type_str.lower().split('|')
        mode_cabin = mode(cabin_types)

        # Cabin type scoring (example - adjust scores as needed)
        if mode_cabin == 'economy':
            cabin_score = 1
        elif mode_cabin == 'premium economy':
            cabin_score = 2
        elif mode_cabin == 'business':
            cabin_score = 3
        elif mode_cabin == 'first':
            cabin_score = 4
        else:
            cabin_score = 0  # Default score for unknown types

        return mode_cabin, cabin_score  # Return both mode and score
    else:
        return 'Unknown', 0


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
    cleaned_df['startingAirport'] = cleaned_df['startingAirport'].astype(str).str.split('|').apply(lambda x: x[0])

    cleaned_df['destinationAirport'] = (
        df['destinationAirport']
        .str.strip()
        .str.upper()
        .str.replace('[^A-Z]', '', regex=True)
    )
    cleaned_df['destinationAirport'] = cleaned_df['destinationAirport'].astype(str).str.split('|').apply(lambda x: x[0])

    cleaned_df['Departure_Year'] = pd.to_datetime(df['flightDate']).dt.year
    cleaned_df['Departure_Month'] = pd.to_datetime(df['flightDate']).dt.month
    cleaned_df['Departure_Day'] = pd.to_datetime(df['flightDate']).dt.day

    df['segmentsDepartureTimeRaw'] = df['segmentsDepartureTimeRaw'].astype(str).str.split('|').apply(lambda x: x[0])
    df['segmentsDepartureTimeRaw'] = pd.to_datetime(df['segmentsDepartureTimeRaw'], utc=True)
    cleaned_df['depatureTimeCategory'] = df['segmentsDepartureTimeRaw'].apply(extract_departure_time)

    cabin_info = df['segmentsCabinCode'].apply(clean_cabin_type)
    cleaned_df['Cabin_Type'] = [cabin[0] for cabin in cabin_info]  # Extract the cabin type (mode)
    cleaned_df['Cabin_Score'] = [cabin[1] for cabin in cabin_info]

    cleaned_df['totalFare'] = df['totalFare']  # Extract the cabin score

    return cleaned_df

# Usage
# Airport_df_cleaned = clean_data(airport_df)