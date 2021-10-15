import pickle
import matplotlib.pyplot as plt

class memory:
    def __init__(self):
        self.distribution_len = {"heavy": {}, "light": {}}
        self.max_seq = {"heavy": 0, "light": 0}
        self.num_rows = 0
        self.corrupted_files = []
        self.normal_files = []

    @staticmethod
    def merge_distribution(x, y):
        z = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
        return z

dest_folder_path = "/Users/hossein/Desktop/log.pkl"
with open(dest_folder_path, 'rb') as inp:
    tech_companies = pickle.load(inp)
    chain_type="light"
    my_dict=tech_companies.distribution_len[chain_type]
    plt.bar(list(my_dict.keys()), my_dict.values(), color='g')
    plt.xlabel("chain length")
    plt.ylabel("Count")
    plt.title("Distribution of {} chains".format(chain_type))
    plt.show()
    print("yes")