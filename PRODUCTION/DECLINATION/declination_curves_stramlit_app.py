import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Función de modelo de declinación exponencial
def exponential_decline(t, qi, D):
    return qi * np.exp(-D * t)

# Modelo de Declinación Hiperbólico
def hyperbolic_decline(t, qi, Di, b):
    return qi / (1 + b * Di * t)**(1/b)

# Modelo de Declinación Armónico
def harmonic_decline(t, qi, Di):
    return qi / (1 + Di * t)

# Carga de datos
uploaded_file = st.file_uploader("Sube tu archivo de producción de petróleo", type=['csv', 'xlsx'])
if uploaded_file is not None:
    try:
        # Determinar el tipo de archivo y cargar los datos
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Procesamiento de datos
        data['DATEPRD'] = pd.to_datetime(data['DATEPRD'])
        data.sort_values('DATEPRD', inplace=True)
        data['Days'] = (data['DATEPRD'] - data['DATEPRD'].min()).dt.days
        data.dropna(subset=['BORE_OIL_VOL'], inplace=True)

        # Visualización de datos brutos
        st.write("Vista Previa de los Datos:", data.head())

        # Preparación de datos para modelado
        t_data = data['Days']
        q_data = data['BORE_OIL_VOL']
        X_train, X_test, y_train, y_test = train_test_split(data[['Days']], q_data, test_size=0.2, random_state=42)

        # Ajuste de modelos de declinación
        popt_exp, _ = curve_fit(exponential_decline, t_data, q_data, bounds=(0, [np.inf, 1]))
        popt_hyp, _ = curve_fit(hyperbolic_decline, t_data, q_data, bounds=(0, [np.inf, 1, 1]))
        popt_harm, _ = curve_fit(harmonic_decline, t_data, q_data, bounds=(0, [np.inf, 1]))

        # Gráfico de los modelos
        t_vals = np.linspace(min(t_data), max(t_data), 500)
        q_pred_exp = exponential_decline(t_vals, *popt_exp)
        q_pred_hyp = hyperbolic_decline(t_vals, *popt_hyp)
        q_pred_harm = harmonic_decline(t_vals, *popt_harm)

        fig, ax = plt.subplots()
        ax.scatter(t_data, q_data, color='blue', label='Datos reales')
        ax.plot(t_vals, q_pred_exp, 'r-', label='Exponencial')
        ax.plot(t_vals, q_pred_hyp, 'g--', label='Hiperbólico')
        ax.plot(t_vals, q_pred_harm, 'm-.', label='Armónico')
        ax.set_title('Curvas de Declinación de Producción')
        ax.set_xlabel('Días')
        ax.set_ylabel('Producción de Petróleo (Sm3)')
        ax.legend()
        st.pyplot(fig)

        # Modelos de ML
        model_linear = LinearRegression()
        model_xgb = XGBRegressor(objective ='reg:squarederror', n_estimators=100)
        model_linear.fit(X_train, y_train)
        model_xgb.fit(X_train, y_train)

        y_pred_linear = model_linear.predict(X_test)
        y_pred_xgb = model_xgb.predict(X_test)

        # Visualización de modelos ML
        fig2, ax2 = plt.subplots()
        ax2.scatter(X_test, y_test, color='gray', label='Datos reales')
        ax2.scatter(X_test, y_pred_linear, color='red', label='Lineal', alpha=0.5)
        ax2.scatter(X_test, y_pred_xgb, color='blue', label='XGBoost', alpha=0.5)
        ax2.set_title('Comparación de Modelos de ML en Curvas de Declinación')
        ax2.set_xlabel('Días')
        ax2.set_ylabel('Producción de Petróleo (Sm3)')
        ax2.legend()
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
