import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

class RecommenderEngine:
    def __init__(self, data):
        self.df = data
        self.user_map = {}
        self.book_map = {}
        self.book_titles = {} # New: Map ISBN to Title
        self.similarity_matrix = None
        
    def prepare_matrix(self):
        print("Mapping IDs and building Sparse Matrix...")
        unique_users = self.df['user_id'].unique()
        unique_books = self.df['isbn'].unique()
        
        self.user_map = {user: i for i, user in enumerate(unique_users)}
        self.book_map = {isbn: i for i, isbn in enumerate(unique_books)}
        
        # Create a lookup for Titles
        # We drop duplicates so we have one title per ISBN
        meta_df = self.df[['isbn', 'title']].drop_duplicates()
        self.book_titles = dict(zip(meta_df['isbn'], meta_df['title']))
        
        self.df['user_idx'] = self.df['user_id'].map(self.user_map)
        self.df['book_idx'] = self.df['isbn'].map(self.book_map)
        
        rows = self.df['user_idx'].values
        cols = self.df['book_idx'].values
        data = self.df['rating'].values
        
        matrix = coo_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_books)))
        self.interaction_matrix = matrix.tocsr()
        
    def train(self):
        print("Training model...")
        item_matrix = self.interaction_matrix.T
        self.similarity_matrix = cosine_similarity(item_matrix, dense_output=False)
        
    def get_recommendations(self, isbn, top_n=5):
        if isbn not in self.book_map:
            return None
        
        book_idx = self.book_map[isbn]
        sim_scores = self.similarity_matrix[book_idx].toarray().flatten()
        top_indices = sim_scores.argsort()[-(top_n+1):][::-1]
        
        results = []
        for idx in top_indices:
            if idx != book_idx:
                # Find ISBN from index, then Title from ISBN
                # (This is a quick way; for production we'd optimize the reverse map)
                found_isbn = list(self.book_map.keys())[list(self.book_map.values()).index(idx)]
                results.append(self.book_titles.get(found_isbn, "Unknown Title"))
                
        return results
