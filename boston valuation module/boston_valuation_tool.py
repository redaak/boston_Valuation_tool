from sklearn.datasets import load_boston as __bs
from sklearn.linear_model import LinearRegression as __rg
from sklearn.metrics import mean_squared_error as __mse
import numpy as np
import pandas as pd

#Loading the dataset
__boston_dataset= __bs()
__data= pd.DataFrame(data=__boston_dataset.data,columns=__boston_dataset.feature_names)

#Transforming data loggin' prices
__features= __data.drop(['INDUS','AGE'], axis=1)
__log_prices=pd.DataFrame(np.log(__boston_dataset.target),columns=['Prices'])

#setting default params for our model as medain of all features
__property_stats= np.ndarray(shape=(1,11))
__property_stats=__features.mean().values.reshape(1,11)

#calculating regrssion
__regr=__rg()
__regration_results=__regr.fit(__features,__log_prices)
__table_fitted_resutls=pd.DataFrame(__regration_results.coef_.reshape(11),columns=['coef'])

#calculating the MSE (mean square error) and RMSE (root mean square error)
__fitted_val=__regration_results.predict(__features)
__Root_mean_squre=np.sqrt(__mse(__log_prices,__fitted_val))

#main dish
def Estimate_Price(numberOfRooms,StudentsPerClass,nextToRiver=False,highConfidance=True):
    """
    ** Estimating the price of a propraty in boston adjusted for inflation **
    Parameters
    ----------
    
    numberOfRooms:  int()
                    number of rooms per house (must be greater or equal to 1)
    
    StudentPerClass: int()
                     the teacher to student ratio per class/ number of student in a class (must be greater or equal to 1)
    
    nextToRiver*: bool() (Default False) 
                  proparty next to the charles river in boston 
    
    highCofidance*: bool() (Default True) 
                    estmation with 95% confidance 
    
    Returns
    -------
    String format with price estamtion:
                predicted price is X $ 
                in the range of -Y $ _ +Z $ 
    
    """
    if numberOfRooms<1 or StudentsPerClass<1:
        print('Impossi a khoya impossi')
        raise ValueError("params entered must be realistic 'be real'")
    #Cpi adjusting for inflation for 2022
    cpi=21.19
    #estimation price model 
    def get_log_estimate(nr_rooms,students_per_class,
                     next_to_river=False,
                     high_confidance=True,
                    ):
            
            #cofiguring params for our linear regression
            __property_stats[0][4]=nr_rooms
            __property_stats[0][8]=students_per_class
            if next_to_river:
                __property_stats[0][2]=1
            else:
                __property_stats[0][2]=0
            log_estimate=__regr.predict(__property_stats)
            if high_confidance:
                    Lower_bound=log_estimate-2*__Root_mean_squre
                    Upper_bound=log_estimate+2*__Root_mean_squre
            else:
                Lower_bound=log_estimate-__Root_mean_squre
                Upper_bound=log_estimate+__Root_mean_squre

            return [log_estimate[0][0],(Lower_bound[0][0],Upper_bound[0][0])]
    #formatin the answer to a readable string
    predicted_price,Range=get_log_estimate(numberOfRooms,StudentsPerClass,nextToRiver,highConfidance)
    lower_bound,upper_bound=Range
    __x="{:,}".format(round(np.e**predicted_price*10000*cpi))
    __z="{:,}".format(round(np.e**lower_bound*10000*cpi))
    __y="{:,}".format(round(np.e**upper_bound*10000*cpi))

    print(f'predicted price is {__x} $ \nin the range of -{__z} $ _ +{__y} $')
    
