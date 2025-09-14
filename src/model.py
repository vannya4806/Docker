from sklearn.linear_model import LogisticRegression

def build_model():
    """
    Membuat dan mengembalikan instance model machine learning.
    Ganti LogisticRegression dengan model lain kalau perlu.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    return model
