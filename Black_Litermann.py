
#########   ###########  ####    ###  |  MODELO BLACK-LITERMANN PARA LA OPTIMIZACION DE PORTAFOLIOS.
########    ##########   ####    ###  |     
###         ###          ####    ###  |  EN ESTE CODIGO USAREMOS IMPLEMENTAREMOS EL MODELO BLACK-
###         #######      ###########  |  LITERMANN PARA LA OPTIMIZACION DE PORTAFOLIOS.
###         ######       ###########  |  
###         ###          ####    ###  |  ADICIONALMENTE, REALIZAREMOS OTRAS OPTIMIZACIONES,
########    ###          ####    ###  |  CALCULAREMOS METRICAS DE RIESGO DE ACTIVOS FINANCIEROS Y
#########   ###          ####    ###  |  REALIZAREMOS BACKTESTING SOBRE LOS PORTAFOLIOS OPTIMIZADOS.   

#---------------------------------------------------------------------------------------------------#
#                                      CARGA DE LIBRERIAS

import pandas as pd
import numpy as np
#from numpy import *
from numpy.linalg import multi_dot
import yfinance as yf
import scipy as stats
from scipy.stats import kurtosis, skew, norm
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns


#---------------------------------------------------------------------------------------------------#
#                                      DESCARGA DE DATOS

# seleccionamos los activos que queremos usar en nuestro portafolio
symbols = ['LIT','IAU','KXI', 'EWW','PE&OLES.MX','ALFAA.MX','ALPEKA.MX', 'IYE']
# definimos la fecha de inicio de los datos
start_date = dt.date(2010, 1, 1)
# definimos la fecha de cierre de los datos
end_date = dt.date.today()

# la siguiente funcion nos ayuda a decargar los datos de cierre cada ticker
def get_asset_data(tickers, start_date, end_date, save_csv = False):
    temp_data = yf.download(tickers, start_date, end_date)["Close"].dropna()
    
    temp_data = temp_data.reset_index()
    
    # Convertir datetime a date
    temp_data['Date'] = pd.to_datetime(temp_data['Date']).dt.date
    temp_data = temp_data.set_index('Date')
    
    return temp_data

data = get_asset_data(symbols, start_date = start_date, end_date = end_date)
print(data.tail())

# la siguiente funcion verifica la moneda de cotizacion de cada ticker
# para efectos de este proyecto, consideraremos como moneda base MXN, por lo que la funcion
# realizara la conversion correspondiente en caso de que el ticker no cotice en MXN
def convert_to_mxn(data, start_date, end_date, target_currency = "MXN"):
    # almacenamos las tasas de cambio
    conversion_rates = {}
    
    for ticker in data.columns:
        symbol = yf.Ticker(ticker)
        currency = symbol.info.get("currency", "USD")  # Por defecto USD si no hay información
        # verificamos si la cotizacion no es en MXN
        if currency != target_currency:
            fx_pair = f"{currency}{target_currency}=X"
    
            # Descargar la tasa de cambio si no ha sido descargada ya
            if fx_pair not in conversion_rates:
                fx_data = pd.DataFrame(get_asset_data(fx_pair, start_date, end_date)).rename( \
                            columns = {"Close" : fx_pair})
                conversion_rates[fx_pair] = fx_data  # Guardar para usos posteriores
    
            # Multiplicamos cada precio del activo por la tasa de cambio de cada dia
            data[ticker] = data[ticker] * conversion_rates[fx_pair][fx_pair]

    return data

target_currency = "MXN"
data = convert_to_mxn(data, start_date, end_date, target_currency)
print(data.tail())


#---------------------------------------------------------------------------------------------------#
#                                GRAFICAS PRECIOS DE CIERRE

# En esta seccion observaremos los precios de cierre de cada uno de los activos considerados

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

for column in data.columns:
    plt.figure()
    plt.plot(data.index, data[column], label=column, color="royalblue", linewidth=1)
    
    # Agregar titulo y etiquetas
    plt.title(f'Valor Historico de {column}', fontsize=16)
    plt.xlabel('Fecha', fontsize=14)
    plt.ylabel(f'Precio de Cierre ({target_currency})', fontsize=14)
    
    # Mejorar el formato de fechas en el eje x
    plt.gcf().autofmt_xdate()

plt.show()


#---------------------------------------------------------------------------------------------------#
#                                     RETORNOS DIARIOS

# definimoa un nuevo data frame con los retornos diarios de cada activo
returns = data.copy()
returns = returns.sort_index()
for columna in returns.columns:
    returns[columna] = (returns[columna] - returns[columna].shift(1)) / returns[columna].shift(1)

returns = returns.dropna()
print(returns.tail())
    

#---------------------------------------------------------------------------------------------------#
#                                   TASAS LIBRE DE RIESGO
    
# la siguiente funcion nos ayudara a obtener la tasa libre de riesgo del bono del tesoro de EUA
# usaremos la API de la FRED para obtener la serie de datos.
# https://fred.stlouisfed.org/docs/api/fred/
# Para obtener una key para la API se debe crear una cuenta de forma gratuita

from fredapi import Fred
# key API FRED
key_fred = '3f2f344c22249ae2ed4577695e869bcd'
# Codigos para bonos del Tesoro en FRED
us_treasury = {"3m":"GS3M", "1y":"GS1", "5y":"GS5", "10y":"GS10"}
plazo = "1y"

# descarga de datos
def get_rf_rate_us(plazo, key, start_date, end_date, today = False):
    fred = Fred(api_key=key)
    rf_rate_us = fred.get_series(plazo, start_date, end_date)
    rf_rate_us = rf_rate_us / 100
    if today == False:
        rf_rate_us = pd.DataFrame(rf_rate_us).rename(columns ={0: "Rate"})
        rf_rate_us.index.name = "Date"
        return rf_rate_us
    else:
        return rf_rate_us[-1]

rf_rate_us = get_rf_rate_us(plazo = us_treasury[plazo], key = key_fred, 
                            start_date = start_date, end_date = end_date)

# vamos a observar la grafica de los resultados
plt.plot(rf_rate_us.index, rf_rate_us["Rate"], color="crimson", linewidth=1)
plt.title(f'{plazo} Treasury Rate US', fontsize=16)
plt.grid(linestyle='--', alpha = 0.7)
plt.show()

us_rf_rate_today = rf_rate_us.iloc[-1][0]
print(f"La tasa libre de riesgo en EU es: {us_rf_rate_today:.4f}")

# Para obtener la tasa libre de riesgo en Mexico usaremos la API de consultas de BANXICO
# esta API es gratuita y tiene un numero de consultas maximo, pero no sera un problema para 
# este estudio
# para mas informacion: https://www.banxico.org.mx/SieAPIRest/service/v1/
import requests
# key API Banxico
key_banxico = "9c64dffdc448adeccfc4ad92a075f06524df61deeae8f2f46206e579f3b2f418"
# codigos para bonos mexicanos
mx_treasury = {"3m": "SF3338", "1y": "SF3367", "5y": "SF18608", "10y": "SF30057"}

def get_rf_rate_mx(plazo, key, start_date, end_date, today = False):
    if today:
        # url para obtener el ultimo dato disponible
        url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{plazo}/datos/oportuno?token={key}"
        response = requests.get(url)
        data = response.json()
        return float(data["bmx"]["series"][0]["datos"][0]["dato"]) / 100  # Convertir a decimal
    else:
        # url para obtener la serie de datos
        url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{plazo}/datos/{start_date}/{end_date}?token={key}"
        response = requests.get(url)
        data = response.json()
        
        # Procesa los datos y los convierte en un DataFrame
        dates = []
        values = []
        for entry in data["bmx"]["series"][0]["datos"]:
            dates.append(entry["fecha"])
            values.append(float(entry["dato"]) / 100)  # Convertir a decimal
        
        # Crear DataFrame con las fechas y los valores
        df = pd.DataFrame({"Date": dates, "Rate": values})
        df["Date"] = pd.to_datetime(df["Date"])  # Convierte las fechas al formato datetime
        df.set_index("Date", inplace=True)
        
        return df
        
rf_rate_mx = get_rf_rate_mx(plazo = mx_treasury[plazo], key = key_banxico,
                            start_date = start_date, end_date = end_date)

# vamos a observar la grafica de los resultados
plt.plot(rf_rate_mx.index, rf_rate_mx["Rate"], color="crimson", linewidth=1)
plt.title(f'{plazo} Treasury Rate MX', fontsize=16)
plt.grid(linestyle='--', alpha = 0.7)
plt.show()

mx_rf_rate_today = rf_rate_mx.iloc[-1][0]
print(f"La tasa libre de riesgo en Mexico es: {mx_rf_rate_today:.4f}")

#---------------------------------------------------------------------------------------------------#
#                            ESTADISTICAS Y METRICAS DE RIESGO

# la siguiente funcion nos ayudara a calcular metricas de riesgo relevantes para el estudio
# de los activos
def metricas(returns, rf_rate):
    """
    Esta funcion calcula estadisticas financieras para una serie de retornos
    Parametros:
        returns (pd.DataFrame): Data Frame con los retornos de los activos a considerar.
                                Cada columna representa un activo
        rf_rate (float): Tasa libre de riesgo
    Returns:
        pd.DataFrame con las estadisticas calculadas de cada activo
    """
    # definimos un diccionario con los resultados
    resultados = {
        'Media': [],
        'Sesgo': [],
        'Exceso de curtosis': [],
        'VaR P 95%': [],
        'VaR H 95%': [],
        'VaR MC 95%': [],
        'CVaR 95%': [],
        'Sharpe R': [],
        'Sortino R': [],
        'Max Drawdown': []
    }
    # consideramos la tasa libre de riesgo diaria
    rf_d = rf_rate / 252
    
    for i in returns.columns:
        mean = np.mean(returns[i])
        stdev = np.std(returns[i])

        # VaR Paramétrico
        pVaR_95 = norm.ppf(1 - 0.95, mean, stdev)

        # VaR Histórico
        hVaR_95 = returns[i].quantile(0.05)

        # VaR Monte Carlo
        n_sims = 1000  # Incrementar el número de simulaciones para mayor precisión
        sim_returns = np.random.normal(mean, stdev, (n_sims, len(returns[i])))
        MCVaR_95 = np.percentile(sim_returns, 5)

        # CVaR
        CVaR_95 = returns[i][returns[i] <= hVaR_95].mean()

        # Sharpe Ratio
        sharpe_ratio = (mean - rf_d) / stdev

        # Sortino Ratio
        neg_returns = returns[i][returns[i] < rf_d]
        sigma_dp = neg_returns.std()
        sortino_ratio = (mean - rf_d) / sigma_dp

        # Max Drawdown
        cumulative_returns = (1 + returns[i]).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        MDD = drawdowns.min()

        # Añadir los resultados al diccionario
        resultados['Media'].append(np.round(np.mean(returns[i]), 6))
        resultados['Sesgo'].append(np.round(skew(returns[i]), 6))
        resultados['Exceso de curtosis'].append(np.round(kurtosis(returns[i], fisher=True), 6))
        resultados['VaR P 95%'].append(np.round(pVaR_95 , 6))
        resultados['VaR H 95%'].append(np.round(hVaR_95 , 6))
        resultados['VaR MC 95%'].append(np.round(MCVaR_95 , 6))
        resultados['CVaR 95%'].append(np.round(CVaR_95 , 6))
        resultados['Sharpe R'].append(np.round(sharpe_ratio, 6))
        resultados['Sortino R'].append(np.round(sortino_ratio, 6))
        resultados['Max Drawdown'].append(np.round(MDD, 6))
    
    # Crear un DataFrame con los resultados
    estadisticas_df = pd.DataFrame(resultados, index=returns.columns)

    return estadisticas_df

metricas_returns = metricas(returns, rf_rate = us_rf_rate_today)
print(metricas_returns)

#---------------------------------------------------------------------------------------------------#
#                                 OPTIMIZACION DE PORTAFOLIOS

import scipy.optimize as sco

# Definimos la funcion portfolio stats para calular retornos, volatilidad y 
# sharpe ratiode los portafolios
def portfolio_stats(weights, returns, return_df = False):
    weights = np.array(weights)[:,np.newaxis]
    port_rets = weights.T @ np.array(returns.mean() * 252)[:,np.newaxis]
    port_vols = np.sqrt(multi_dot([weights.T, returns.cov() * 252, weights]))
    sharpe_ratio = port_rets/port_vols
    resultados = np.array([port_rets, port_vols, sharpe_ratio]).flatten()
    
    if return_df == True:
        return pd.DataFrame(data = np.round(resultados,4),
                            index = ["Returns", "Volatility", "Sharpe_Ratio"],
                            columns = ["Resultado"])
    else:
        return resultados

# Definimos las fechas sobre las que queremos optimizar el portafolio
#start_date_opt = min(returns.index)
#end_date_opt = max(returns.index)

start_date_opt = dt.date(2016, 1, 1)
end_date_opt = dt.date(2022, 12, 31)

# Guardamos los retornos en un nuevo df
returns1 = returns.loc[start_date_opt:end_date_opt]


## 1. MINIMA VOLATILIDAD ---------------------------------------------------------------------------

# definimos la funcion que nos ayudara a obtener la volatilidad del portafolio con portfolio_stats
def get_volatility(weights, returns):
    return portfolio_stats(weights, returns)[1]

# vamos a definir una funcion que optimice el portafolio bajo minima volatilidad
def min_vol_opt(returns):
    # Definimos las condiciones para nuestra optimizacion
        # la suma de los activos debe ser 1
        # el peso de cada activo debe estar entre 0 y 1
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    bnds = tuple((0, 1) for x in range(len(returns.columns)))
    
    # usaremos un portafolio equaly weighted como pesos iniciales
    initial_wts = np.array(len(returns.columns)*[1./len(returns.columns)])
    
    # Usamos la funcion minimizar de scipy
    opt_vol = sco.minimize(fun=get_volatility, x0=initial_wts, args=(returns),
                           method='SLSQP', bounds=bnds, constraints=cons)
    
    # obtenemos los pesos del portafolio bajo la optimizacion
    #min_vol_pesos = list(zip(returns.columns, np.around(opt_vol['x']*100,2)))
    min_vol_pesos = pd.DataFrame(data = np.around(opt_vol['x']*100,2),
                                 index = returns.columns,
                                 columns = ["Min_Vol"]) 
    
    # obtenemos las estadisticas del portafolio optimizado
    min_vol_stats = portfolio_stats(opt_vol['x'], returns, return_df = True)
    min_vol_stats = min_vol_stats.rename(columns={"Resultado":"Min_Vol"})
    
    return {"min_vol_pesos": min_vol_pesos, "min_vol_stats": min_vol_stats}
    
min_vol_resultados = min_vol_opt(returns1)

# los pesos del portafolio optimizado con minima varianza son
print(min_vol_resultados["min_vol_pesos"])

# las estadisticas del portafolio optimizado son
print(min_vol_resultados["min_vol_stats"])


## 2. MAX Sharpe Ratio -----------------------------------------------------------------------------

# ya que en esta optimizacion buscamos maximizar el sharpe ratio, definiremos una funcion
# de apoyo que obtendra el valor negativo del sharpe ratio calculado en la funcion
# portfolio stats para poder usar la funcion de miniminzacion de scipy
def min_sharpe_ratio(weights, returns):
    return -portfolio_stats(weights, returns)[2]

#type(min_sharpe_ratio(min_vol_resultados["min_vol_pesos"]["Min_Vol"]/100, returns1))
#min_sharpe_ratio(min_vol_resultados["min_vol_pesos"]["Min_Vol"]/100, returns1)

# definimos la funcion que hara la optimizacion
def max_sr_opt(returns):
    # Definimos las condiciones para nuestra optimizacion
        # la suma de los activos debe ser 1
        # el peso de cada activo debe estar entre 0 y 1
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    bnds = tuple((0, 1) for x in range(len(returns.columns)))
    
    # usaremos un portafolio equaly weighted como pesos iniciales
    initial_wts = np.array(len(returns.columns)*[1./len(returns.columns)])
    
    # Usamos la funcion minimizar de scipy
    opt_sr = sco.minimize(fun=min_sharpe_ratio, x0=initial_wts, args=(returns),
                           method='SLSQP', bounds=bnds, constraints=cons)
    
    # obtenemos los pesos del portafolio bajo la optimizacion
    #min_vol_pesos = list(zip(returns.columns, np.around(opt_vol['x']*100,2)))
    max_sr_pesos = pd.DataFrame(data = np.around(opt_sr['x']*100,2),
                                 index = returns.columns,
                                 columns = ["Max_SR"]) 
    
    # obtenemos las estadisticas del portafolio optimizado
    max_sr_stats = portfolio_stats(opt_sr['x'], returns, return_df = True)
    max_sr_stats = max_sr_stats.rename(columns={"Resultado":"Max_SR"})
    
    return {"max_sr_pesos": max_sr_pesos, "max_sr_stats": max_sr_stats}

max_sr_resultados = max_sr_opt(returns1)

# los pesos del portafolio optimizado con minima varianza son
print(max_sr_resultados["max_sr_pesos"])

# las estadisticas del portafolio optimizado son
print(max_sr_resultados["max_sr_stats"]) 
    

## 3. Minima Volatilidad con Objetivo de Rendimiento -----------------------------------------------

# definimos la funcion para optimizar el portafolio
def min_vol_obj_opt(returns, r_obj):
    # definimos las condiciones para la optimizacion
    cons = ({'type': 'eq', 'fun': lambda x: portfolio_stats(x, returns)[0] - r_obj},
                   {'type': 'eq', 'fun': lambda x: sum(x) - 1})
    bnds = tuple((0, 1) for x in range(len(returns.columns)))
    
    # usaremos un portafolio equaly weighted como pesos iniciales
    initial_wts = np.array(len(returns.columns)*[1./len(returns.columns)])
    
    # Usamos la funcion minimizar de scipy
    opt_min_obj = sco.minimize(fun=get_volatility, x0=initial_wts, args=(returns),
                           method='SLSQP', bounds=bnds, constraints=cons)
    
    # obtenemos los pesos del portafolio bajo la optimizacion
    #min_vol_pesos = list(zip(returns.columns, np.around(opt_vol['x']*100,2)))
    min_obj_pesos = pd.DataFrame(data = np.around(opt_min_obj['x']*100,2),
                                 index = returns.columns,
                                 columns = ["Min_Vol_Obj"]) 
    
    # obtenemos las estadisticas del portafolio optimizado
    min_obj_stats = portfolio_stats(opt_min_obj['x'], returns, return_df = True)
    min_obj_stats = min_obj_stats.rename(columns={"Resultado":"Min_Vol_Obj"})
    
    return {"min_obj_pesos": min_obj_pesos, "min_obj_stats": min_obj_stats}

min_obj_resultados = min_vol_obj_opt(returns1, r_obj = 0.10)

# los pesos del portafolio optimizado con minima varianza son
print(min_obj_resultados["min_obj_pesos"])

# las estadisticas del portafolio optimizado son
print(min_obj_resultados["min_obj_stats"]) 


## Resultados --------------------------------------------------------------------------------------

# comparamos los pesos de cada activo en los portafolios
resultados_pesos = min_vol_resultados["min_vol_pesos"].merge(max_sr_resultados["max_sr_pesos"],
                    on = "Ticker").merge(min_obj_resultados["min_obj_pesos"], on = "Ticker")
print(resultados_pesos)

# comparamos las metricas de cada portafolio
resultados_stats = min_vol_resultados["min_vol_stats"].join(max_sr_resultados["max_sr_stats"])\
                    .join(min_obj_resultados["min_obj_stats"])
print(resultados_stats)


#---------------------------------------------------------------------------------------------------#
#                                      BACKTESTING

# Realizaremos backtesging de los portafolios optimizados en el punto anterior y compararemos
# los resultados con el S&P 500

# Definimos las fechas sobre las que realizaremos el backtesting
start_bt = dt.date(2021, 1, 1)
end_bt = dt.date(2024, 1, 1)
# seleccionamos los retornos entre las fechas seleccionadas
returns_bt = returns.loc[start_bt:end_bt]

# descargamos los datos del S&P 500
sp500 = get_asset_data("^GSPC", start_bt, end_bt).rename(columns = {'Close':'^GSPC'})
# conversion de divisas
sp500 = convert_to_mxn(sp500, start_bt, end_bt)
# obtenemos los retornos diarios
returns_sp500 = sp500.copy().sort_index()
for columna in returns_sp500.columns:
    returns_sp500[columna] = (returns_sp500[columna] - returns_sp500[columna].shift(1)) / returns_sp500[columna].shift(1)

returns_sp500 = returns_sp500.dropna()


## 1. Retornos Anuales

# copiamos los retornos para obtener los datos anuales
returns_bt_y = returns_bt.copy()
# Asegurarse de que el índice sea de tipo datetime
returns_bt_y.index = pd.to_datetime(returns_bt_y.index)
# obtenemos los retornos anuales de cada activo
returns_bt_y = (returns_bt_y.resample('Y').apply(lambda x: (1 + x).prod() - 1))
#returns_bt_y = returns_bt.resample('Y').mean()*252

# obtenemos los retornos anuales de cada portafolio usando los pesos de portafolios optimizados
for i in range(1, len(returns_bt_y) + 1):
    sub_returns = returns_bt_y.iloc[:i].T

    returns_min_vol = np.dot(np.array(min_vol_resultados["min_vol_pesos"]).T, sub_returns)
    returns_sr = np.dot(np.array(max_sr_resultados["max_sr_pesos"]).T, sub_returns)
    returns_min_obj = np.dot(np.array(min_obj_resultados["min_obj_pesos"]).T, sub_returns)
    returns_ew = np.dot(np.array(len(returns_bt_y.columns)*[100./len(returns_bt_y.columns)]).T, sub_returns)
    
resultados_bt_y = {"Minima Volatilidad": returns_min_vol.tolist()[0],
                 "Maximo Sharpe Ratio": returns_sr.tolist()[0],
                 "Minima Vol. Objetivo": returns_min_obj.tolist()[0],
                 "Equaly Weighted": returns_ew.tolist()}
resultados_bt_y = pd.DataFrame.from_dict(resultados_bt_y).set_index(returns_bt_y.index.year)    
   
# hacemos lo mismo proceso con los datos del S&P 500
returns_sp500_y = returns_sp500.copy()
returns_sp500_y.index = pd.to_datetime(returns_sp500_y.index)
returns_sp500_y = (returns_sp500_y.resample('Y').apply(lambda x: (1 + x).prod() - 1))*100
returns_sp500_y =returns_sp500_y.set_index(returns_sp500_y.index.year).rename(columns = {"^GSPC": "S&P 500"})

# agregamos los datos al dataframe
resultados_bt_y = resultados_bt_y.join(returns_sp500_y)

# graficamos los resultados
fig, ax = plt.subplots(figsize=(10, 6))
# Numero de anios (indice) y numero de portafolios (columnas)
years = resultados_bt_y.index
categories = resultados_bt_y.columns
num_categories = len(categories)
width = 0.12  # Ancho de cada barra

# Crear un conjunto de barras para cada activo por año
for i, category in enumerate(categories):
    ax.bar(
        years + i * width,  # Desplazamos las barras para cada portafolio
        resultados_bt_y[category], 
        width=width,
        label=category
    )

# Añadir etiquetas y leyendas
ax.set_ylabel("Rendimiento Anual (%)")
ax.set_title("Rendimiento Anual por Portafolio")
ax.set_xticks(years + width * (num_categories - 1) / 2)
ax.set_xticklabels(years)
ax.legend(title = "Portafolio")
ax.grid(axis='y', linestyle='--', alpha=0.7)
    
# Mostrar el gráfico
plt.show()


## 2. Comportamiento diario

# obtenemos los retornos anuales de cada portafolio usando los pesos de portafolios optimizados
for i in range(1, len(returns_bt) + 1):
    sub_returns = returns_bt.iloc[:i].T
    returns_min_vol_d = np.dot(np.array(min_vol_resultados["min_vol_pesos"]).T, sub_returns)
    returns_sr_d = np.dot(np.array(max_sr_resultados["max_sr_pesos"]).T, sub_returns)
    returns_min_obj_d = np.dot(np.array(min_obj_resultados["min_obj_pesos"]).T, sub_returns)
    returns_ew_d = np.dot(np.array(len(returns_bt_y.columns)*[100./len(returns_bt_y.columns)]).T, sub_returns)


resultados_bt_d = {"Minima Volatilidad": (returns_min_vol_d.tolist()[0]),
                 "Maximo Sharpe Ratio": returns_sr_d.tolist()[0],
                 "Minima Vol. Objetivo": returns_min_obj_d.tolist()[0],
                 "Equaly Weighted": returns_ew_d.tolist()}
resultados_bt_d = pd.DataFrame.from_dict(resultados_bt_d).set_index(returns_bt.index) 

# dividimos los retornos entre 100
resultados_bt_d = resultados_bt_d / 100
# agregamos los retornos del SP 500
resultados_bt_d = resultados_bt_d.join(returns_sp500).fillna(0)



# definimos una inversion inicial en el portafolio
inversion_inicial = 1000
# obtenemos el valor de cada portafolio a traves del tiempo
valor_portafolio = inversion_inicial * (1 + resultados_bt_d).cumprod()
    
# graficamos los resultados
fig, ax = plt.subplots(figsize=(10, 6))
# Numero de anios (indice) y numero de portafolios (columnas)
years = resultados_bt_y.index
categories = resultados_bt_y.columns
num_categories = len(categories)

# Crear un conjunto de barras para cada activo por año
for i in valor_portafolio.columns:
    plt.plot(valor_portafolio.index, valor_portafolio[i], label=f'{i}', linewidth = 1)
plt.title(f"Valor diario del portafolio con inversión inicial de: ${inversion_inicial} {target_currency}")
plt.xlabel("Fecha")
plt.ylabel(f"Valor del Portafolio {target_currency}")
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.show()
    
# observemos las estadisticas del backtesting
bt_stats = metricas(resultados_bt_d, rf_rate = us_rf_rate_today)
print(bt_stats)

#---------------------------------------------------------------------------------------------------#
#                                      MODELO BLACK-LITERMANN

# Para el siguiente punto debemos obtener las tasas libre de riesgo vigentes al inicio
# de cada anio sobre el que tenemos informacion
# Usaremos las tasas anuales vigentes al inicio de cada periodo

rates = []
# ingrese la tasa que quiere considerar
#pais = "MX"
pais = "US"
if plazo != "1y":
    if pais == "MX":
        temp_rate = get_rf_rate_mx(mx_treasury["1y"], key_banxico, start_date, end_date)
        # nos quedamos con las tasas vigentes el 01 de enero
        for y in range(start_date.year, end_date.year + 1):
            rates.append(temp_rate.loc[f"{y}-01-01"])
    elif pais == "US":
        temp_rate = get_rf_rate_us(us_treasury["1y"], key_fred, start_date, end_date)
        # nos quedamos con las tasas vigentes el 01 de enero
        for y in range(start_date.year, end_date.year + 1):
            rates.append(temp_rate.loc[f"{y}-01-01"])
else:
    if pais == "MX":
        for y in range(start_date.year, end_date.year + 1):
            rates.append(rf_rate_mx.loc[f"{y}-01-01"])
    elif pais == "US":
        for y in range(start_date.year, end_date.year + 1):
            rates.append(rf_rate_us.loc[f"{y}-01-01"])

# transformamos la lista a un DataFrame
rf_rates = pd.DataFrame(rates)
rf_rates = rf_rates.set_index(rf_rates.index.year)
rf_rates.index.name = "Year"
print(rf_rates)

# ahora vamos a obtener los retornos acumulados anuales de cada activo
annual_returns = returns.copy()
# modificamos el indice a datetime
annual_returns.index = pd.to_datetime(annual_returns.index)
# obtenemos los retornos anuales
annual_returns = annual_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
# modificamos el indice para conservar solo los anios
annual_returns = annual_returns.set_index(annual_returns.index.year)

# conservamos las tasas libres de riesgo correspondientes
rf_rates = rf_rates.loc[annual_returns.index]

# ahora obtenemos el exceso de retorno
excess_returns = annual_returns.subtract(rf_rates.iloc[:, 0], axis=0)

# obtenemos la matriz de varianzas y covarianzas de los excesos de retorno
cov_matrix = excess_returns.cov()
print(cov_matrix)

# para calcular la distribucion a priori del modelo, asumimos un benchmark constituido por
# los activos seleccionados con un peso equitativo
ew_pesos =  len(cov_matrix) * [1./len(cov_matrix)]
ew_pesos = np.array(ew_pesos)[:,np.newaxis]

# calculamos la desviacion estandar
desv_est_bl = np.sqrt(ew_pesos.T @ cov_matrix @ ew_pesos)
print(desv_est_bl)

# definimos Lambda
Lambda = (1/desv_est_bl)*0.5
print(Lambda)

# distribucion a priori
vec_ec_bl = (cov_matrix @ ew_pesos) @ Lambda
print(vec_ec_bl)

# definimos Tau: 1/(numero de periodos (anios))
Tau = 1/annual_returns.shape[0]
print(Tau)

# varianza a priori
var_priori = Tau * cov_matrix
print(var_priori)

## Introducir views:
    
# Ahora introduciremos nuestras views abosolutas y relativas sobre cada activo:
# Ej. view absoluta: El activo 3 tendra un rendimiento de 15%
# Ej. view relativa: El activo 2 tendra un rendimiento 10% al rendimiento del activo 1

# debemos definir 3 matrices:
#   1. Views
#   2. Retorno esperado sobre las views
#   3. Confianza sobre la View

print("Ejemplo de matriz de Views con 4 activos y 3 views:")
print(np.array([[0,0,0,1],[1,-1,0,0],[0,0,1,0]]))
print("En este caso tenemos una view absoluta sobre el activo 4,")
print("una view relativa del activo 1 respecto al activo 2")
print("y una view absoluta sobre el activo 3.")
print("---"*30)

print(f"La matriz de views debe tener dimensiones {cov_matrix.shape[0]} x n.")
print("n representa el numero de views.")

print("Posicion de los activos:")
for i, j in zip(cov_matrix.columns, range(1, cov_matrix.shape[1] + 1)):
    print(f"Activo {j}: {i}")
    
input("Presione enter para continuar:")   
P = np.array([
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [-1, 1, 0, 0, 0 ,0 ,1 ,0],
    [0, 0, 0, 0, 1, 0, 0, 0]
    ])

print("Ejemplo de matriz de retorno esperado con 4 activos y 3 views:")
print(np.array([[0.2],[0.1],[0.16]]))
print("En este caso esperamos que el activo 4 tenga un retorno del 20%,")
print("el activo 1 tenga un retorno 10% mayor al activo 2")
print("y el activo 3 tenga un rendimiento del 16%.")
print("---"*30)

print(f"La matriz de retornos debe tener dimensiones 1 x {P.shape[0]}.")

input("Presione enter para continuar:")   
Q = np.array([[0.20], [0.10], [0.10], [0.15], [0.12]])

print("Ejemplo de matriz de cofianza con 4 activos y 3 views:")
print(np.array([0.5, 0.6, 0.7]))
print("En este caso tenemos 50% de confianza sobre la view 1,")
print("60% sobre la view 2 y 70% sobre la view 3.")
print("---"*30)

print(f"La matriz de confianza ser una matriz diagonal con dimensiones {P.shape[0]} x {P.shape[0]}.")

input("Presione enter para continuar:")   
O = np.diag([0.40, 0.4, 0.6, 0.15, 0.2])

# definimos una matriz auxiliar definida como views @ var_priori @ views.T 
aux1 = np.array(P @ var_priori @ P.T)

# obtenemos la matriz diagonalizada
O2 = np.diag(np.diag(aux1))

# obtenemos la esperanza del exceso de retorno
E = np.linalg.inv(np.linalg.inv(Tau*cov_matrix) + P.T@(np.linalg.inv(O2)@P)) \
    @ (np.linalg.inv(Tau*cov_matrix)@vec_ec_bl + P.T@np.linalg.inv(O2)@Q)

# varianza de los estimados
varianza_E = np.linalg.inv(np.linalg.inv(Tau*cov_matrix)+ P.T@(np.linalg.inv(O2)@P))

## Portafolio Optimizado Black-Litermann

# Lambda representa el caso base de adversion al riesgo al obtener la distribucion
# a priori. Considerando otros valores de Lambda, podemos obtener los pesos del 
# portafolio optimizado condistintos niveles de riesgo.
Lambda1 = Lambda.iloc[0,0]
# Entre menor sea el valor de Lambda, se toma mas riesgo.
# Un valor mayor representa mas ADVERSION al riesgo.
list_lambda = np.arange(1, 7.0, 0.25).tolist()
list_lambda.insert(0, Lambda1.round(4))

portafolios_bl = pd.DataFrame()
# observemos los pesos del portafolio optimizado con distintos valores de riesgo
for i in list_lambda:
    weights_bl = (np.linalg.inv(cov_matrix * i)) @ E
    portafolios_bl = pd.concat([portafolios_bl, pd.DataFrame(weights_bl).T.rename(index={0:i})])

portafolios_bl.index.name = "Lambda"
portafolios_bl.columns = cov_matrix.columns
portafolios_bl['Total'] = portafolios_bl.sum(axis=1)
print(portafolios_bl)













    