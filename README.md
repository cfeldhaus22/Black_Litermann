# Análisis y Optimización de Portafolios Financieros con Python

Este proyecto es una aplicación en Python para descargar, analizar y optimizar datos financieros de activos. Permite transformar activos a una moneda específica, calcular métricas de riesgo y rendimiento, optimizar portafolios y realizar un backtesting comparativo con el índice S&P 500. Además, incluye la implementación del modelo Black-Litterman para personalizar las optimizaciones en función de "views" financieras.

## Características del Proyecto

### 1. Descarga de Datos Financieros
- **Fuente de Datos**: Descarga datos históricos de activos financieros usando la API de Yahoo Finance.
- **Conversión de Moneda**: Realiza la conversión de activos a una moneda específica (ej. USD a MXN) para una comparación consistente entre activos.

### 2. Visualización de Precios
- **Gráficos de Precios de Cierre**: Visualización de precios históricos para cada activo.
- **Retornos Diarios**: Cálculo y visualización de los retornos diarios para analizar la volatilidad.

### 3. Tasas Libres de Riesgo
- **Fuentes de Tasas de Interés**: Integra las APIs de Banxico y FRED para obtener tasas libres de riesgo.

### 4. Cálculo de Métricas de Riesgo y Rendimiento
- **Valor en Riesgo (VaR)**: Estimación del riesgo máximo en un periodo dado bajo condiciones normales.
- **Exceso de Curtosis**: Análisis del riesgo de eventos extremos en los activos.
- **Sortino Ratio**: Métrica de rendimiento ajustada por el riesgo de retornos negativos.
- **Sharpe Ratio**: Relación rendimiento-riesgo ajustada por la tasa libre de riesgo.

### 5. Optimización de Portafolios
Optimización basada en:
- **Mínima Volatilidad**: Portafolio con menor volatilidad.
- **Máximo Sharpe Ratio**: Portafolio con mejor relación rendimiento-riesgo.
- **Mínima Volatilidad con Rendimiento Objetivo**: Portafolio con volatilidad mínima para un rendimiento específico.

### 6. Backtesting de Portafolios
- **Comparación con S&P 500**: Se realiza backtesting para evaluar la efectividad de las estrategias generadas, comparándolas contra el rendimiento del índice S&P 500.

### 7. Modelo Black-Litterman
- **Views Financieras Personalizadas**: Permite ajustar la optimización del portafolio en función de las expectativas de rendimiento de cada activo mediante el modelo Black-Litterman.

## Tecnologías y Librerías

- **Python**: Lenguaje de desarrollo principal.
- **Yahoo Finance API**: Descarga de datos de mercado.
- **APIs de Banxico y FRED**: Tasas de interés sin riesgo.
- **Pandas**: Manipulación de datos.
- **NumPy**: Operaciones numéricas.
- **Matplotlib y Seaborn**: Visualización de datos financieros.
- **Scipy**: Herramientas de optimización.

## Contribución

Si tienes ideas o mejoras para este proyecto, ¡serán bienvenidas! Puedes crear un fork de este repositorio y abrir un Pull Request con tus cambios.

## Licencia

Este proyecto es de uso público y educativo. Los datos utilizados son de acceso público.

---

Cualquier duda o comentario sobre el análisis o el código, ¡no dudes en contactarme!
