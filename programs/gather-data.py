
import pandas as pd
import boto3
import json
import os
import gzip
import datetime
import pytz


s3 = boto3.client('s3')
bucket_name = 'colorado-catholic-buisness-directory'

print('Getting list of all CSV files in data folder')
# Get list of all CSV files in data folder
data_files = []
data_folder = 'data'  # Data folder relative to project root
for file in os.listdir(data_folder):
    if file.endswith('.csv'):
        data_files.append(os.path.join(data_folder, file))

# Read and concatenate all CSV files
dfs = []
for file in data_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Combine all dataframes
if dfs:
    df = pd.concat(dfs, ignore_index=True)
    # Get the latest date
    latest_date = df['originalTimestamp'].max()
    print(f"Latest date in data: {latest_date}")
else:
    print("No CSV files found in data folder")

print('Getting list of all files in S3')
s3_folder = '/segment-logs/bdy6BRcrFwgvPGEtNFNwSA/'
files = []
file_dates = []
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder.lstrip('/'))

for page in pages:
    for obj in page.get('Contents', []):
        files.append(obj['Key'])
        file_dates.append(obj['LastModified'])

# Create a dictionary mapping filenames to their upload dates
file_info = dict(zip(files, file_dates))

print('Downloading and importing data from S3')
def download_import_data(file_name, file_date):
    # Skip if file date is not newer than latest_date
    latest_date_formatted = datetime.datetime.strptime(latest_date.split()[0], '%Y-%m-%d')
    latest_date_formatted = latest_date_formatted.replace(tzinfo=pytz.UTC)
    
    if file_date <= latest_date_formatted:
        return pd.DataFrame()
        
    local_file_name = os.path.basename(file_name)
    with open(local_file_name, 'wb') as f:
        s3.download_fileobj(bucket_name, file_name, f)

    # Read the gzipped JSON file into a DataFrame
    # Segment typically stores data as newline-delimited JSON (NDJSON)
    df = pd.DataFrame()

    with gzip.open(local_file_name, 'rt') as f:
        # Read file line by line since each line is a separate JSON object
        data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)

    # Expand the context column into separate columns
    context_df = pd.json_normalize(df['context'])
    # Drop the original context column and join with expanded columns
    df = df.drop('context', axis=1).join(context_df, rsuffix='_context')
    try:
        properties_df = pd.json_normalize(df['properties'])
        df = df.drop('properties', axis=1).join(properties_df, rsuffix='_properties')
    except:
        pass

    os.remove(local_file_name)

    return df

print('Concatenating data from S3')
full_df = pd.DataFrame()

for file, file_date in file_info.items():
    df = download_import_data(file, file_date)
    full_df = pd.concat([full_df, df])

print(f'Total rows in full dataframe: {full_df.shape[0]}')



# Fill missing values in 'event' column with values from 'type' column
full_df['event_type'] = full_df['event'].fillna(full_df['type'])


# Convert originalTimestamp to datetime
full_df['originalTimestamp'] = pd.to_datetime(full_df['originalTimestamp'])

# remove columns that are not useful for analysis
full_df.drop(columns=['projectId', '_metadata', 'sentAt', 'version', '__segment_internal', 'locale', 'timezone', 'ip', 
                      'library.version', 'timestamp', 'channel', 'integrations', 'event', 'type', 'library.name', 'receivedAt'], inplace=True)


full_df.reset_index(drop=True, inplace=True)
print(f'Total rows in full dataframe: {full_df.shape[0]}')

# Get current datetime for filename
current_time = pd.Timestamp.now().strftime('%Y-%m-%d')
filename = f'data/ccbd-{current_time}.csv'

full_df.to_csv(filename, index=False)
print(f'Saved full dataframe to {filename}')





