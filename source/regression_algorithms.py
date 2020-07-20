from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

REGRESSION = {"RFR": RandomForestRegressor(n_estimators=100),
              "KNNR": KNeighborsRegressor(),
              "RNR": RadiusNeighborsRegressor(),
              "MLPR": MLPRegressor(),
              "SVR": SVR(),
              "SGDR": SGDRegressor(),
              "ABR": AdaBoostRegressor(),
              "GBR": GradientBoostingRegressor()
              }

classifiers_list = ['RFR', 'KNNR', 'RNR', 'MLPR', 'SVR', 'SGDR', 'ABR', 'GBR']
