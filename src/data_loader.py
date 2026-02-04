import pandas as pd
import os
import boto3
from io import StringIO


class DataLoader:
    def __init__(self, file_path, use_s3=False, bucket_name=None):
        """
        Initializes the data loader.
        :param file_path: Local path or S3 key.
        :param use_s3: Boolean flag to switch between Local and AWS S3.
        :param bucket_name: AWS S3 bucket name (required if use_s3 is True).
        """
        self.file_path = file_path
        self.use_s3 = use_s3
        self.bucket_name = bucket_name

    def load_data(self):
        """
        Extracts data from Source (Local or AWS S3).
        """
        if self.use_s3:
            print(
                f"Downloading {self.file_path} from S3 Bucket: {self.bucket_name}...")
            s3 = boto3.client('s3')
            obj = s3.get_object(Bucket=self.bucket_name, Key=self.file_path)
            data = obj['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(data), delimiter=';',
                             on_bad_lines='skip', encoding='latin-1')
        else:
            print(f"Loading data from local path: {self.file_path}...")
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found at {self.file_path}")
            df = pd.read_csv(self.file_path, delimiter=';',
                             on_bad_lines='skip', encoding='latin-1')

        return self.clean_data(df)

    def clean_data(self, df):
        """
        Transforms data: Cleans nulls and ensures correct data types.
        """
        print("Cleaning and transforming data...")

        # Rename columns for consistency
        df = df.rename(columns={'User-ID': 'user_id',
                       'ISBN': 'book_id', 'Rating': 'rating'})

        # Ensure rating is numeric and drop invalid rows
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating'])

        # Filter for implicit ratings (0 often means interaction but no rating)
        # For this engine, we might only want explicit ratings > 0
        df = df[df['rating'] > 0]

        return df
