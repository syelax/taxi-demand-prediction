from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR


def download_file(year: int, month: int) -> Path:
    """
    Downloads a parquet file with historical taxi rides for a given month and year.

    """
    URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"

    resp = requests.get(URL)

    # check to see if get request was sucessful
    if resp.status_code == 200:

        path = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        with open(path, 'wb') as f:
            f.write(resp.content)
        
        return path
    else:
        raise Exception(f'{URL} is not available')
    

def validate_raw_data(
        rides: pd.DataFrame,
        year: int,
        month: int,
) -> pd.DataFrame:
    """ 
    Removes rows with pickup_datetimes outside the valid range 
    """

    this_month = f'{year}-{month:02d}-01'
    next_month = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    date_filter_mask = (rides.pickup_datetime >= this_month) & (rides.pickup_datetime < next_month)
    return rides[date_filter_mask]

def load_raw_data(
        year: int ,
        months: Optional[List[int]] = None
) -> pd.DataFrame:
    
    rides = pd.DataFrame()

    if months is None:
        # download all months in year
        months = list(range(1,13))
    elif isinstance(months, int):
        months = [months]

    for month in months:

        local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        if not local_file.exists():
            try:
                print(f'Downloading file {year}-{month:02d}')
                download_file(year, month)
            except:
                print(f'{year}-{month:02d} file is not available')
                continue

        else:
            print(f'File {year}-{month:02d} was already available in local storage.')

        
        rides_one_month = pd.read_parquet(local_file)

        # rename columns
        rides_one_month = rides_one_month[['tpep_pickup_datetime', 'PULocationID']]
        rides_one_month.rename(columns={
            'tpep_pickup_datetime':'pickup_datetime', 
            'PULocationID':'pickup_location_id',
        }, inplace=True)

        # validate raw data 
        rides_one_month = validate_raw_data(rides_one_month, year, month)

        # append to existing data 
        rides = pd.concat([rides, rides_one_month])

    return rides[['pickup_datetime', 'pickup_location_id']]
        
        
def transform_raw_data_into_ts_data(
        rides: pd.DataFrame,
) -> pd.DataFrame:
    
    # get hourly rides for each pickup location
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('h')
    agg = rides.groupby(by=['pickup_hour', 'pickup_location_id']).size().reset_index()
    agg.rename(columns={0:'rides'}, inplace=True)

    agg_rides_all_slots = add_missing_timeslots(agg)

    return agg_rides_all_slots

def add_missing_timeslots(agg_rides: pd.DataFrame) -> pd.DataFrame:

    locations = agg_rides['pickup_location_id'].unique()
    full_range = pd.date_range(
        agg_rides['pickup_hour'].min(), agg_rides['pickup_hour'].max(), freq='h'
    )
    output = pd.DataFrame()
    print(agg_rides.columns)
    for location in tqdm(locations):
    
        agg_rides_i = agg_rides.loc[agg_rides.pickup_location_id == location, ['pickup_hour', 'rides']]

        agg_rides_i.set_index('pickup_hour', inplace=True)
        agg_rides_i.index = pd.DatetimeIndex(agg_rides_i.index)
        agg_rides_i = agg_rides_i.reindex(full_range, fill_value=0)

        agg_rides_i['pickup_location_id'] = location

        output = pd.concat([output, agg_rides_i])

    return output.reset_index().rename(columns={'index':'pickup_hour'})
    
def get_cutoff_indices(
        data: pd.DataFrame,
        n_features: int,
        step_size: int
) -> list:
    stop_position = len(data) - 1

    subseq_first_idx = 0
    subseq_mid_idx = n_features
    subseq_last_idx = n_features + 1
    indices = []

    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))

        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size
    
    return indices

def transform_ts_data_into_features_and_target(
        ts_data: pd.DataFrame,
        input_seq_len: int,
        step_size: int,
)-> pd.DataFrame:
    """
    Slices and transposes data from time-series format into a (features, target)
    format that we can use to train Supervised ML Models
    """
    assert set(ts_data.columns) == {'pickup_hour', 'rides', 'pickup_location_id'}

    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    for location_id in tqdm(location_ids):
        
        # keep only ts data for this `location_id`
        ts_data_one_location = ts_data.loc[
            ts_data.pickup_location_id == location_id, 
            ['pickup_hour', 'rides']
        ].sort_values(by=['pickup_hour'])

        # pre-compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices(
            ts_data_one_location,
            input_seq_len,
            step_size
        )

        # slice and transpose data into numpy arrays for features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        pickup_hours = []
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values[0]
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        # numpy -> pandas
        features_one_location = pd.DataFrame(
            x,
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id

        # numpy -> pandas
        targets_one_location = pd.DataFrame(y, columns=[f'target_rides_next_hour'])

        # concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets['target_rides_next_hour']