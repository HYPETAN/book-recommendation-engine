from src.data_loader import DataLoader
from src.recommender import RecommenderEngine
import time

def main():
    # CONFIGURATION
    USE_S3 = False
    BUCKET_NAME = "your-bucket-name"
    
    # We now look for TWO files
    RATINGS_FILE = "data/Ratings.csv" 
    BOOKS_FILE = "data/Books.csv"

    # 1. Load Data
    loader = DataLoader(RATINGS_FILE, BOOKS_FILE, use_s3=USE_S3, bucket_name=BUCKET_NAME)
    
    try:
        df = loader.load_data()
        print(f"Data Loaded Successfully. Rows: {len(df)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you created a 'data' folder and put both .csv files inside!")
        return

    # 2. Initialize Engine
    engine = RecommenderEngine(df)
    engine.prepare_matrix()
    
    # 3. Train Model
    start_time = time.time()
    engine.train()
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # 4. Generate Recommendation
    # Test with a random book from the merged data
    test_isbn = df['isbn'].iloc[0] 
    test_title = df['title'].iloc[0]
    
    print(f"\nGenerating recommendations for: '{test_title}'")
    
    recs = engine.get_recommendations(test_isbn)
    
    if recs:
        print("\nTop 5 Recommended Books:")
        for i, book_title in enumerate(recs):
            print(f"{i+1}. {book_title}")
    else:
        print("Book not found.")

if __name__ == "__main__":
    main()
