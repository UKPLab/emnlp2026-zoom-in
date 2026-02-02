import pickle

def serialized_size_mb(obj) -> float:
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)) / (1024 ** 2)
