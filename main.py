from src.data_loader import DataLoader
from src.recommender import RecommenderEngine
import time


def main():
    # CONFIGURATION
    # To use AWS, set USE_S3 to True and provide bucket name
    USE_S3 = False
    BUCKET_NAME = "your-s3-bucket-name"
    FILE_PATH = "data/Ratings.csv"  # Local path or S3 key

    # 1. Load Data
    loader = DataLoader(FILE_PATH, use_s3=USE_S3, bucket_name=BUCKET_NAME)
    df = loader.load_data()
    print(f"Data Loaded. Rows: {len(df)}")

    # 2. Initialize Engine
    engine = RecommenderEngine(df)
    engine.prepare_matrix()

    # 3. Train Model
    start_time = time.time()
    engine.train()
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # 4. Generate Recommendation
    # Let's pick a random book from the dataset to test
    test_book = df['book_id'].iloc[0]
    print(f"\nGenerating recommendations for Book ISBN: {test_book}")

    recs = engine.get_recommendations(test_book)

    if recs:
        print("Top 5 Recommended Books:")
        for i, book in enumerate(recs):
            print(f"{i+1}. {book}")
    else:
        print("Book not found in database.")


if __name__ == "__main__":
    main()
