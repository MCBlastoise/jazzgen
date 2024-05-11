import pickle

def access_pickle_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def store_pickle_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)