import time
from oversampling import Oversampling

def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:0.1f}".format(int(hours), int(minutes), seconds))


def main():
    start = time.time()
    print('INIT')
    folder_experiments = './../datasets/'
    dtosmoter = Oversampling()
    print('STEP 1')
    #dtosmoter.createValidationData(folder_experiments)
    print('STEP 2')
    #dtosmoter.runSMOTEvariationsGen(folder_experiments)
    print('STEP 3')
    #dtosmoter.runDelaunayVariationsGen(folder_experiments)
    print('STEP 4')
    dtosmoter.runRegression(folder_experiments)

    end = time.time()
    print("Total Execution Time : ")
    timer(start, end)


if __name__ == "__main__":
    main()