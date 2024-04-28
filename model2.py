#import necessary libraries of python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option("plotting.backend", "plotly")

# Load the dataset
data = pd.read_csv('Diamonds/data_diamonds.csv')
print(data)

data['Price'] = data['Price'].str.replace(',', '').astype(np.float64)

# Function to extract all the numbers from the given string
def getNumbers(str):
    import re
    
    array = re.findall(r'[0-9]', str)
    return array

data['Mesurements'] = data['Mesurements'].apply(lambda x: getNumbers(x) )
data['Mesurements'] = data['Mesurements'].apply(lambda x: ''.join(x) )

data['length']= data['Mesurements'].str[:3].astype(np.float64) /100
data['width'] = data['Mesurements'].str[3:6].astype(np.float64) /100
data['depth'] = data['Mesurements'].str[6:].astype(np.float64) / 100


print('\033[1m' + 'Shape of the data :' + '\033[0m')
print(data.shape, 
          '\n------------------------------------------------------------------------------------\n')
print('\033[1m' + 'All columns from the dataframe :' + '\033[0m')
print(data.columns, 
          '\n------------------------------------------------------------------------------------\n')
print('\033[1m' + 'Datatpes and Missing values:' + '\033[0m')
print(data.info(), 
          '\n------------------------------------------------------------------------------------\n')   
print('\033[1m' + 'Missing value count:' + '\033[0m')
print(data.isnull().sum(),
          '\n------------------------------------------------------------------------------------\n')
print('\033[1m' + 'Summary statistics for the data' + '\033[0m')
print(data.describe(include='all'), 
          '\n------------------------------------------------------------------------------------\n')
print('\033[1m' + 'Memory used by the data :' + '\033[0m')
print(data.memory_usage(), 
          '\n------------------------------------------------------------------------------------\n')
print('\033[1m' + 'Number of duplicate values :' + '\033[0m')
print(data.duplicated().sum(),
          '\n------------------------------------------------------------------------------------\n')
print('\033[1m' + 'Outliers in the data :' + '\033[0m')

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)

IQR = Q3-Q1
outliers = (data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))
print(outliers.sum()) 

del data['Mesurements']
print("---------------data-------------------")
# print(data.loc[619])
print(data)

#Feature Analysis
features = data[ 
    ['Weight',
     'Shape',
     'Clarity',
     'Colour',
     'Cut',
     'Polish',
     'Symmetry',
     'Fluorescence',
     'length',
     'width',
     'depth',
     ]
]

# print(features )

labels= data['Price']

X_train, X_test, y_train, y_test = train_test_split(pd.get_dummies( features ), labels, test_size=0.0002, random_state=0)


X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

print("--------X_train-----------")
print(X_train)




def make_corr_map(data, title, zmin=-1, zmax=1, height=600, width= 800):
    """
    data: Your dataframe.
    title: Title for the correlation matrix.
    zmin: Minimum number for color scale. (-1 to 1). Default = -1.
    zmax: Maximum number for color scale. (-1 to 1). Default = 1.
    height: Default = 600
    width: Default = 800
    """
    
    data = data.corr()
    mask = np.triu(np.ones_like(data, dtype=bool))
    rLT = data.mask(mask)
    heat = go.Heatmap(
        z = rLT,
        x = rLT.columns.values,
        y = rLT.columns.values,
        zmin = zmin, 
            # Sets the lower bound of the color domain
        zmax = zmax,
            # Sets the upper bound of color domain
        xgap = 1, # Sets the horizontal gap (in pixels) between bricks
        ygap = 1,
        colorscale = 'RdBu'
    )

    title = title

    layout = go.Layout(
        title_text=title, 
        title_x=0.5, 
        width= width, 
        height= height,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
    )

    fig= go.Figure([heat], layout=layout)
    return fig.show()


Xy_train = pd.concat([X_train, y_train], axis=1)
make_corr_map(Xy_train, 'Cool title', height=1500, width=1500)

gbr = GradientBoostingRegressor(n_estimators=100)
rfr = RandomForestRegressor(n_estimators=100,oob_score=True)

rfr_model = rfr.fit(X_train,y_train)
gbr_model = gbr.fit(X_train, y_train)

plt.figure(figsize= (18, 45))

feature_importance = gbr.feature_importances_
indices = np.argsort(feature_importance)

plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
plt.barh(range(len(indices)), feature_importance[indices], color='m', align='center')
plt.show()


# Make predictions for the test set
gbr_pred = gbr_model.predict(X_test)
rfr_pred = rfr_model.predict(X_test)



#Gradient Boosting Regressor
print("For gbr :")

# View accuracy score
print('Accuracy for Train (gbr):', gbr.score(X_train, y_train) )
print('Accuracy for Test (gbr):', gbr.score(X_test, y_test) )


pred_resg =pd.DataFrame({'Actual':y_test, 'Predicted(gbr)':gbr_pred})

print(pred_resg)

# Print out the mean absolute error for gbr(mae)
print('Mean Absolute Error (gbr):', round( metrics.mean_absolute_error(y_test, gbr_pred),2 ))
print('Mean Squared Error (gbr):', round( metrics.mean_squared_error(y_test, gbr_pred), 2))

# Calculate mean absolute percentage error (MAPE)
errorsg = abs(gbr_pred - y_test)
mapeg = 100 * (errorsg / y_test)

# Calculate and display accuracy
accuracyg = 100 - np.mean(mapeg)
print('Accuracy:', round(accuracyg, 2), '%.')



#Random forest regressor
print("For rfr :")


# View accuracy score
print('Accuracy for Train (rfr):', rfr.score(X_train, y_train) )
print('Accuracy for Test (rfr):', rfr.score(X_test, y_test) )


pred_resr =pd.DataFrame({'Actual':y_test, 'Predicted(rfr)':rfr_pred})

print(pred_resr)

# Print out the mean absolute error for rfr(mae)
print('Mean Absolute Error (rfr):', round( metrics.mean_absolute_error(y_test, rfr_pred),2 ))
print('Mean Squared Error (rfr):', round( metrics.mean_squared_error(y_test, rfr_pred), 2))

# Calculate mean absolute percentage error (MAPE)
errorsr = abs(rfr_pred - y_test)
maper = 100 * (errorsr / y_test)

# Calculate and display accuracy
accuracyr = 100 - np.mean(maper)
print('Accuracy:', round(accuracyr, 2), '%.')


# pickle.dump(rfr_model, open('model.pkl', 'wb'))