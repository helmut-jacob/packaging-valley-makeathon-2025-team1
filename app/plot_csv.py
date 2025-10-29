import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import csv
from datetime import datetime
import sys



matplotlib.use('WebAgg')







def main():
    filename = sys.argv[1]
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
        rows = [row for row in reader]
        print(rows)
        timestamps, _, _, values = zip(*rows)
        print(timestamps)
        print(values)
        plt.plot([datetime.fromisoformat(ts) for ts in timestamps], [float(value) if value else np.nan for value in values])
        plt.show()




if __name__ == "__main__":
    main()
