import time

from graphs import Performance
from oversampling import Oversampling
from parameters import output_dir, result_dir, folder_experiments


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:0.1f}".format(int(hours), int(minutes), seconds))


def main():
    start = time.time()
    print('INIT')
    
    #dtosmoter = Oversampling()
    #print('STEP 1')
    #dtosmoter.createValidationData(folder_experiments)
    #print('STEP 2')
    #dtosmoter.runSMOTEvariationsGen(folder_experiments)
    #print('STEP 3')
    #dtosmoter.runDelaunayVariationsGen(folder_experiments)
    #print('STEP 4')
    #dtosmoter.runRegression(folder_experiments)
    
    r = 'v1'
    
    analisys = Performance()
    #analisys.average_results(output_dir+'results_regression.csv',r)
    #analisys.run_rank_choose_parameters(result_dir+'regression_average_results_' + r + '.csv', release=r)
    #analisys.grafico_variacao_alpha(r)
    analisys.find_best_rank(output_dir,r)
    
    

    end = time.time()
    print("Total Execution Time : ")
    timer(start, end)


if __name__ == "__main__":
    main()