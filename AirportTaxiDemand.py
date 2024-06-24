

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import adfuller
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor




try:
    df = pd.read_csv(
    '/home/rosypc2/Escritorio/SP13/taxi.csv', index_col=[0], parse_dates=[0])
except:
    df = pd.read_csv("/datasets/taxi.csv") 



#Ordenamos los datos de acuerdoa al indice
df= df.sort_index()

# %%
#Remmuestramos a 1 hora 
df = df.resample('1H').sum()



#Graficando los datos
plt.figure(figsize=(12, 6))
plt.title('Time series of taxi demand')
plt.plot(df.index, df['num_orders'], label='Number of orders')
plt.legend()
plt.show()


# Para suavizar la información y observar mejor el comportamiento vamos a crear las graficas de tendencia y estacionalidad  para los ultimos 30 dias, y para las ultimas 48 horas. 


decomposed = seasonal_decompose(df['num_orders'][-720:])

plt.figure(figsize=(12, 12))
plt.subplot(311)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residual')
plt.tight_layout()

# %%
decomposed = seasonal_decompose(df['num_orders'][-48:])

plt.figure(figsize=(12, 6))
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')
plt.tight_layout()

# %%
# vamos a realizar la prueba "ADF test" (Augmented Dickey-Fuller test), esta es una prueba estadística utilizada para evaluar si una serie temporal es estacionaria o no. La estacionariedad es una propiedad importante en el análisis de series temporales y significa que las propiedades estadísticas de la serie no cambian con el tiempo.
adft = adfuller(df, autolag="AIC")

# Imprimir el valor p
print('Valor p:', adft[1])

# Comprobar la significancia
if adft[1] < 0.05:
    print('La serie temporal es estacionaria.')
else:
    print('La serie temporal no es estacionaria.')


# Ya hemos determinado que nuestra serie es estacionario, veamoslo graficamente en un periodo mas largo, tambien hemos visto que hay un compartamiento ciclico por dia

# %%
# Features of the date
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek

# Features of a 24-hour time lag
df['lag_24'] = df['num_orders'].shift(24)


# Features of a 24-hour rolling mean
df['rolling_mean'] = df['num_orders'].shift().rolling(24).mean()


# %%
plt.figure(figsize=(12, 6))
plt.title('Time series of taxi demand with additional features')
plt.plot(df.index[-168:], df['num_orders'][-168:], label='Number of orders')
plt.plot(df.index[-168:], df['lag_24'][-168:], label='24-hour lag')
plt.plot(df.index[-168:], df['rolling_mean'][-168:], label='24-hour rolling mean')
plt.legend()
plt.show()





# asegurando de no tener valores ausentes
df=df.dropna()
df.info()


# ## Construccion del modelo

# %%
random_state = 42
features = df.drop('num_orders', axis=1)
target = df['num_orders']

# Dividiendo los datos segun los solicitado training and test sets (90% y 10%)
features_train, features_test, target_train, target_test = train_test_split(features, target, shuffle=False, test_size=0.2, random_state=random_state)


# En virtud de que se esta trabajandpo en un problema de regresion, se establece como metrica de puntuacion en 'neg_mean_squared_error',y se busca minimizar el error cuadrático medio (MSE).

# %%
scoring = 'neg_mean_squared_error'

# %%
#Crear la función para definir los mejores hiperparametros para la lista de modelos
def find_best_params(models, features_train, target_train, scoring):
    # Crear el DataFrame para almacenar los resultados
    results = pd.DataFrame(columns=['Model', 'Best Parameters', 'Best Score'])

    # Iterar en la lista de modelos
    for model in models:
        # Imprimir el nombre del modelo
        print(f"Finding best parameters for {type(model['model']).__name__}...")

        # Grid search de los hiperparametros
        grid = GridSearchCV(model['model'], model['param_grid'], cv=5, scoring=scoring, verbose=0, n_jobs=-1)
        grid.fit(features_train, target_train)

        # Extraer los mejores parametros y score
        best_params = grid.best_params_
        best_score = np.abs(grid.best_score_)

        # Almacenar los resultados en el DataFrame
        results = pd.concat([results, pd.DataFrame({'Model': type(model['model']).__name__, 
                                  'Best Parameters': [best_params], 
                                  'Best Score': best_score})], ignore_index=True)

        # Ordenando de acuerdo a la mejor puntuación
        results.sort_values(by='Best Score', ascending=False, inplace=True)
        
    return results

# %%
#Definir los modelos y sus hiperparametros
models = [
    {
        'model': LinearRegression(),
        'param_grid': {}
    },
    {
        'model': RandomForestRegressor(random_state= random_state),
        'param_grid': {'n_estimators': np.arange(10, 30, 50), 'max_depth': np.arange(3, 5)}
    },
    {
        'model': CatBoostRegressor(random_state=random_state, silent=True),
        'param_grid':  {'iterations': np.arange(10, 30, 50), 'depth': np.arange(3, 5)}
    },
    {
        'model': xgb.XGBRegressor(random_state= random_state),
        'param_grid': {'n_estimators': np.arange(10, 50, 30), 'max_depth': np.arange(3, 5)}
    },
     {
        'model': lgb.LGBMRegressor(random_state= random_state),
        'param_grid': {'n_estimators': np.arange(10, 50, 30), 'max_depth': np.arange(3, 5)}
    },
]
# Encontrar el mejor hiperparametro para cada modelo
results = find_best_params(models, features_train, target_train, scoring)
results
   


# Se utiliza la forma negativa del MSE en la búsqueda de hiperparámetros o la validación cruzada, esto significa que penaliza más los errores más grandes y por tanto busca minimizar su valor.
# En nuestra busqueda que encontramos que el mejor "Best_score" es el valor mas alto, en este caso al modelo de regression Lineal.
# 

# %%
# Estableciendo el índice del Dataframe al nombre del modelo
results.set_index('Model', inplace=True)

# Estableciendo los hiperparémetros para cada modelo según los resulotados de la busqueda de la cuadricula (grid search).
models = [
    LinearRegression(**results.loc['LinearRegression']['Best Parameters']),
    RandomForestRegressor(random_state=42, **results.loc['RandomForestRegressor']['Best Parameters']),
    CatBoostRegressor(random_state=42, **results.loc['CatBoostRegressor']['Best Parameters'], silent=True),
    xgb.XGBRegressor(random_state=42, **results.loc['XGBRegressor']['Best Parameters']),
    lgb.LGBMRegressor(random_state=42, **results.loc['XGBRegressor']['Best Parameters'])
]


# ## Entrenando y evaluando el modelo

# %%
# Creando una funcion para entrenar y evaluar los diferentes modelos 
def train_and_evaluate_models(models, features_train, target_train, features_valid, target_valid):
    # Creando un  DataFrame para almacenar los resultados
    results = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'R2', 'RMSE'])
    
    # Loop through each model
    for model in models:
        # Train the model
        model.fit(features_train, target_train)
        
        # Make predictions
        predictions = model.predict(features_valid)
        
        # Evaluate the model
        mae = mean_absolute_error(target_valid, predictions)
        mse = mean_squared_error(target_valid, predictions)
        r2 = r2_score(target_valid, predictions)
        rmse = np.sqrt(mse)
        
        # Append the results to the DataFrame
        results = pd.concat([results, pd.DataFrame({
            'Model': model.__class__.__name__, 'MAE': mae, 'MSE': mse, 'R2': r2, 'RMSE': rmse}, index=[0])], ignore_index=True)
    
    return results


# Entrenar y evaluar los modelos
train_results = train_and_evaluate_models(models, features_train, target_train, features_test, target_test)

# Print the results
train_results


# Hemos logrado evaluar y entrenar 5 diferentes modelos para predecir la cantidad de pedidos de taxis en la proxima hora. Previemente realizamos una análisis exploratorio de datos centrandonos en la variable num_orders (Número de pedidos) y re-muestreamos los datos para mejorar su calidad.
# Tambien mejoramos la precisión de los modelos mediant el ajuste de hiperparametros.
# El único modelo que nos arrojó un valor menor a 48 fue el de Regresion Lineal, en segundo lugar y superando minimamente el valor de 48 establecido como maximo fue el de CatBoostRegresor.
# En resumen, hemos desarrollado un modelo que puede predecir con suficiente precisión el número de pedidos de taxis en la próxima hora, proporcionando una herramienta valiosa para Sweet Lift Taxi.


