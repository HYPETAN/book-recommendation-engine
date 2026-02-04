import pandas as pd
import os
import boto3
from io import StringIO

class DataLoader:
    def __init__(self, ratings_path, books_path, use_s3=False, bucket_name=None):
        """
        Initializes the data loader.
        :param ratings_path: Path to Ratings.csv
        :param books_path: Path to Books.csv
        """
        self.ratings_path = ratings_path
        self.books_path = books_path
        self.use_s3 = use_s3
        self.bucket_name = bucket_name

    def load_data(self):
        """
        Loads and merges Ratings and Books data.
        """
        print("Loading datasets...")
        
        # Load Ratings
        # dtype={'ISBN': str} ensures we don't lose leading zeros (e.g. '019...')
        ratings = self._read_csv(self.ratings_path, names=['user_id', 'isbn', 'rating'])
        
        # Load Books (Only need ISBN and Title)
        books = self._read_csv(self.books_path, names=['isbn', 'title', 'author', 'year', 'publisher'])
        
        # Merge them
        print("Merging datasets...")
        # We merge on ISBN to attach Titles to the Ratings
        df = pd.merge(ratings, books[['isbn', 'title']], on='isbn', how='inner')
        
        return self.clean_data(df)

    def _read_csv(self, path, names):
        """Helper to read CSV from Local or S3"""
        # We skip the first row (header=0) because we provided manual 'names' above
        if self.use_s3:
            s3 = boto3.client('s3')
            obj = s3.get_object(Bucket=self.bucket_name, Key=path)
            data = obj['Body'].read().decode('utf-8')
            return pd.read_csv(StringIO(data), delimiter=';', on_bad_lines='skip', encoding='latin-1', dtype={'isbn': str}, header=0, names=names)
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            return pd.read_csv(path, delimiter=';', on_bad_lines='skip', encoding='latin-1', dtype={'isbn': str}, header=0, names=names)

    def clean_data(self, df):
        """
        Transforms data: Cleans nulls and ensures correct data types.
        """
        # Ensure rating is numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating', 'title']) # Drop if no rating or no title
        
        # Filter for explicit ratings (optional, usually rating > 0)
        df = df[df['rating'] > 0]
        
        return df
