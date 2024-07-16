import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
import matplotlib.pyplot as plt

# Verifica se o ambiente tem suporte para exibição gráfica
try:
    import matplotlib.pyplot as plt
    print("Matplotlib backend:", plt.get_backend())
except ImportError as e:
    print(e)
    print("Please install matplotlib using `pip install matplotlib`")

# Dados dos resultados mensais até Abril de 2024
data = {
    'ano/mês': pd.date_range(start='2022-01-01', periods=30, freq='MS'),
    'resultado': [
        0.87119, 0.58079, 0.58079, 0.87119, 1.45199, 1.45199, 0.87119, 1.74239, 0.87119, 0.87119, 0, 1.45199,
        2.03279, 0.58079, 2.61358, 1.16159, 1.16159, 1.74239, 0.58079, 2.32318, 1.45199, 2.03279, 0.58079, 1.45199, 1.74239, 0.29040, 2.03279, 3.19, 3.78, 3.19
    ]
}

# Metas para 2024
metas_2024 = [2.00, 0.57, 3.15, 1.14, 1.14, 1.72, 0.57, 2.86, 1.43, 2.57, 0.86, 1.14]

# Criar DataFrame
df = pd.DataFrame(data)
df.set_index('ano/mês', inplace=True)
df.index.freq = 'MS'

# Modelo de Suavização Exponencial Tripla com componentes aditivos
model_hw = ExponentialSmoothing(df['resultado'], trend='add', seasonal='add', seasonal_periods=12)
fit_hw = model_hw.fit()

# Previsões para o futuro
previsoes_hw = fit_hw.forecast(steps=20)

# Defina o nível de confiança desejado (por exemplo, 99% de confiança)
confidence_level = 0.9
z = norm.ppf((1 + confidence_level) / 2)

# Calcular intervalo de confiança para Holt-Winters
sigma_hw = np.std(fit_hw.resid, ddof=1)  # desvio padrão dos resíduos

df_previsao_hw = pd.DataFrame({
    'previsão': previsoes_hw,
    'limite_superior': previsoes_hw + z * sigma_hw,
    'limite_inferior': previsoes_hw - z * sigma_hw
}, index=pd.date_range(start=df.index[-1] + pd.offsets.MonthEnd(1), periods=20, freq='MS'))

# Adicionar metas ao DataFrame
df_meta = pd.DataFrame({
    'meta': metas_2024
}, index=pd.date_range(start='2024-01-01', periods=12, freq='MS'))

# Função para ajustar modelo ARIMA e prever
def ajustar_modelo_arima(series):
    model_arima = ARIMA(series, order=(6,1,0))
    model_fit_arima = model_arima.fit()
    forecast_arima = model_fit_arima.forecast(steps=20)
    return forecast_arima, model_fit_arima

previsoes_arima, fit_arima = ajustar_modelo_arima(df['resultado'])

# Calcular intervalo de confiança para ARIMA
sigma_arima = np.std(fit_arima.resid, ddof=1)  # desvio padrão dos resíduos

df_previsao_arima = pd.DataFrame({
    'previsão': previsoes_arima,
    'limite_superior': previsoes_arima + z * sigma_arima,
    'limite_inferior': previsoes_arima - z * sigma_arima
}, index=pd.date_range(start=df.index[-1] + pd.offsets.MonthEnd(1), periods=20, freq='MS'))

# Filtra apenas os dados de 2024 para plotagem
df_2024 = df.loc['2024']
df_previsao_hw_2024 = df_previsao_hw.loc['2024']
df_previsao_arima_2024 = df_previsao_arima.loc['2024']
df_meta_2024 = df_meta.loc['2024']

# Ajustar um polinômio aos pontos previstos pelo ARIMA
x_vals = np.arange(len(df_previsao_arima_2024))
y_vals = df_previsao_arima_2024['previsão'].values
p = np.polyfit(x_vals, y_vals, 3)  # Ajusta um polinômio de grau 3
y_fit = np.polyval(p, x_vals)
equation = f'${p[0]:.2e}x^3 + {p[1]:.2e}x^2 + {p[2]:.2e}x + {p[3]:.2e}$'

# Gráfico de Holt-Winters
plt.figure(figsize=(14, 7))
plt.plot(df_2024.index, df_2024['resultado'], label='Resultado Real', marker='o')
for x, y in zip(df_2024.index, df_2024['resultado']):
    plt.text(x, y, f'{y:.2f}', fontsize=9, ha='right')

plt.plot(df_previsao_hw_2024.index, df_previsao_hw_2024['previsão'], label='Previsão Suavizada', linestyle='--', marker='o')
for x, y in zip(df_previsao_hw_2024.index, df_previsao_hw_2024['previsão']):
    plt.text(x, y, f'{y:.2f}', fontsize=9, ha='right')

plt.plot(df_meta_2024.index, df_meta_2024['meta'], label='Meta 2024', linestyle=':', color='red', marker='x')
for x, y in zip(df_meta_2024.index, df_meta_2024['meta']):
    plt.text(x, y, f'{y:.2f}', fontsize=9, ha='right')

plt.fill_between(df_previsao_hw_2024.index, df_previsao_hw_2024['limite_superior'], df_previsao_hw_2024['limite_inferior'], color='grey', alpha=0.2)

plt.title(f'Resultados e Previsões com Suavização Tripla Aditiva e Metas\n({int(confidence_level*100)}% Intervalo de Confiança)')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico ARIMA
plt.figure(figsize=(14, 7))
plt.plot(df_2024.index, df_2024['resultado'], label='Resultado Real', marker='o')
for x, y in zip(df_2024.index, df_2024['resultado']):
    plt.text(x, y, f'{y:.2f}', fontsize=9, ha='right')

plt.plot(df_previsao_arima_2024.index, df_previsao_arima_2024['previsão'], label='Previsão ARIMA', linestyle='--', marker='o')
for x, y in zip(df_previsao_arima_2024.index, df_previsao_arima_2024['previsão']):
    plt.text(x, y, f'{y:.2f}', fontsize=9, ha='right')

plt.plot(df_meta_2024.index, df_meta_2024['meta'], label='Meta 2024', linestyle=':', color='red', marker='x')
for x, y in zip(df_meta_2024.index, df_meta_2024['meta']):
    plt.text(x, y, f'{y:.2f}', fontsize=9, ha='right')

plt.fill_between(df_previsao_arima_2024.index, df_previsao_arima_2024['limite_superior'], df_previsao_arima_2024['limite_inferior'], color='grey', alpha=0.2)

# Plotar a linha de ajuste polinomial
plt.plot(df_previsao_arima_2024.index, y_fit, label=f'Ajuste Polinomial: {equation}', linestyle='-', color='blue')

plt.title(f'Previsões com ARIMA e Limites de Confiança\n({int(confidence_level*100)}% Intervalo de Confiança)')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.grid(True)
plt.show()

# Exportar para CSV
df_complete_hw = pd.concat([df, df_previsao_hw], axis=0)
df_complete_hw['meta'] = df_meta['meta']
df_complete_hw.to_csv('resultados_HW_aditivada.csv', decimal=',')

df_complete_arima = pd.concat([df, df_previsao_arima], axis=0)
df_complete_arima['meta'] = df_meta['meta']
df_complete_arima.to_csv('resultados_previsoes_arima.csv', decimal=',')
