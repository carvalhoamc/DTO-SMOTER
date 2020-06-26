from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

classifiers = {"RF": RandomForestRegressor(n_estimators=100),
               "KNN": KNeighborsRegressor(),
               "RNR": RadiusNeighborsRegressor(),
               "LR": LinearRegression(),
               "LGR": LogisticRegression(),
               "ISOR": IsotonicRegression(),
               "MLP": MLPRegressor(),
               "SVR":SVR(),
               "SGD":SGDRegressor(),
               "ABR":AdaBoostRegressor(),
               "GBR":GradientBoostingRegressor()
               }

classifiers_list = ['RF', 'KNN', 'RNR', 'LR', 'LGR', 'ISOR', 'MLP', 'SVR','SGD','ABR']
