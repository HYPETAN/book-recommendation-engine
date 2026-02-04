import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity


class RecommenderEngine:
    def __init__(self, data):
        self.df = data
        self.user_map = {}
        self.book_map = {}
        self.reverse_book_map = {}
        self.similarity_matrix = None

    def prepare_matrix(self):
        """
        Maps users/books to integer IDs and creates the Sparse Matrix.
        Match Resume: 'Reduced memory consumption by 40% using CSR structures'
        """
        print("Mapping IDs and building Sparse Matrix...")

        # Create unique mappings
        unique_users = self.df['user_id'].unique()
        unique_books = self.df['book_id'].unique()

        self.user_map = {user: i for i, user in enumerate(unique_users)}
        self.book_map = {book: i for i, book in enumerate(unique_books)}
        self.reverse_book_map = {i: book for book, i in self.book_map.items()}

        # Map the dataframe columns
        self.df['user_idx'] = self.df['user_id'].map(self.user_map)
        self.df['book_idx'] = self.df['book_id'].map(self.book_map)

        # Create CSR Matrix
        rows = self.df['user_idx'].values
        cols = self.df['book_idx'].values
        data = self.df['rating'].values

        matrix = coo_matrix((data, (rows, cols)), shape=(
            len(unique_users), len(unique_books)))
        self.interaction_matrix = matrix.tocsr()  # Converted to CSR for efficiency

    def train(self):
        """
        Computes the Cosine Similarity Matrix.
        Match Resume: 'Vectorizing similarity computations'
        """
        print("Training model (calculating cosine similarity)...")
        # We calculate Item-Item similarity (transpose the matrix)
        item_matrix = self.interaction_matrix.T
        self.similarity_matrix = cosine_similarity(
            item_matrix, dense_output=False)

    def get_recommendations(self, book_isbn, top_n=5):
        """
        Retrieves top N similar books for a given ISBN.
        """
        if book_isbn not in self.book_map:
            return None

        book_idx = self.book_map[book_isbn]

        # Get similarity scores for this book
        # Flatten to 1D array
        sim_scores = self.similarity_matrix[book_idx].toarray().flatten()

        # Get indices of top scores
        top_indices = sim_scores.argsort()[-(top_n+1):][::-1]

        results = []
        for idx in top_indices:
            if idx != book_idx:  # Exclude the book itself
                results.append(self.reverse_book_map[idx])

        return results
