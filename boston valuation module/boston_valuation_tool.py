
class EvalTool:
    from sklearn.datasets import load_boston as __bs
    from sklearn.linear_model import LinearRegression as __rg
    from sklearn.metrics import mean_squared_error as __mse
    import numpy as __np
    import pandas as __pd
    def __init__(self):
        #loading the data set for the sklearn toydataset
        self.__boston_dataset= __bs()
        self.__data= pd.DataFrame(data=self.__boston_dataset.data,columns=self.__boston_dataset.feature_names)
        #transforming Data loggin prices to adjust for the skew
        self.__features= self.__data.drop(['INDUS','AGE'], axis=1)
        slef.__log_prices=__pd.DataFrame(np.log(self.__boston_dataset.target),columns=['Prices'])
        
        #Creating default argumment for our model
        self.__property_stats=__np.ndarray(shape=(1,11))
        self.__property_stats=self.__features.mean().values.reshape(1,11)
        # calulating the regession using Sklearn module
        self.__regr=__rg()
        self.__regration_results=regr.fit(self.__features,log_prices)
        table_fitted_resutls=__pd.DataFrame(regration_results.coef_.reshape(11),columns=['coef'])
        
        # calculating the mean square error and the root square error
        fitted_val=regration_results.predict(self.__features)
        Root_mean_squre=np.sqrt(mse(self.__log_prices,fitted_val))
        # calling and formating the answer
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
            #estimation price model 
            def get_log_estimate(nr_rooms,students_per_class,
                             next_to_river=False,
                             high_confidance=True,
                            ):
                    #Cpi adjusting for inflation
                    cpi=21.19
                    #cofiguring params for our linear regression
                    self.__property_stats[0][4]=nr_rooms
                    self.__property_stats[0][8]=students_per_class
                    if next_to_river:
                        self.__property_stats[0][2]=1
                    else:
                        self.__property_stats[0][2]=0
                    self.__log_estimate=regr.predict(property_stats)
                    if high_confidance:
                            Lower_bound=log_estimate-2*Root_mean_squre
                            Upper_bound=log_estimate+2*Root_mean_squre
                    else:
                        Lower_bound=log_estimate-Root_mean_squre
                        Upper_bound=log_estimate+Root_mean_squre

                    return [log_estimate[0][0],(Lower_bound[0][0],Upper_bound[0][0])]
            #formatin the answer to a readable string
            predicted_price,Range=self.__get_log_estimate(numberOfRooms,StudentsPerClass,nextToRiver,highConfidance)
            lower_bound,upper_bound=Range
            print(f'predicted price is {round(np.e**predicted_price*10000*cpi)}$ \nin the range of -{round(np.e**lower_bound*10000*cpi)}$ _ +{round(np.e**upper_bound*10000*cpi)}$')
