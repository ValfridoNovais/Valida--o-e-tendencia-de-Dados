import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Dados dos resultados mensais até Abril de 2024
data = {
    'ano/mês': pd.date_range(start='2022-01-01', periods=29, freq='MS'),
    'resultado': [
        0.87119, 0.58079, 0.58079, 0.87119, 1.45199, 1.45199, 0.87119, 1.74239, 0.87119, 0.87119, 0, 1.45199,
        2.03279, 0.58079, 2.61358, 1.16159, 1.16159, 1.74239, 0.58079, 2.32318, 1.45199, 2.03279, 0.58079, 1.45199,
        1.74239, 0.29040, 1.74239, 3.48478, 2.0328
    ]
}

# Metas para 2024
metas_2024 = [2.00, 0.57, 3.15, 1.14, 1.14, 1.72, 0.57, 2.86, 1.43, 2.57, 0.86, 1.14]

# Criar DataFrame
df = pd.DataFrame(data)
df.set_index('ano/mês', inplace=True)
df.index.freq = 'MS'

# Modelo de Suavização Exponencial Tripla com componentes aditivos
model = ExponentialSmoothing(df['resultado'], trend='add', seasonal='add', seasonal_periods=12)
fit = model.fit()

# Previsões para o futuro
previsoes = fit.forecast(steps=20)
df_previsao = pd.DataFrame({
    'previsão': previsoes
}, index=pd.date_range(start=df.index[-1] + pd.offsets.MonthEnd(1), periods=20, freq='MS'))

# Adicionar metas ao DataFrame
df_meta = pd.DataFrame({
    'meta': metas_2024
}, index=pd.date_range(start='2024-01-01', periods=12, freq='MS'))

# Gráfico
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['resultado'], label='Resultado Real', marker='o')
for x, y in zip(df.index, df['resultado']):
    plt.text(x, y, f'{y:.2f}', fontsize=9, ha='right')

plt.plot(df_previsao.index, df_previsao['previsão'], label='Previsão Suavizada', linestyle='--', marker='o')
for x, y in zip(df_previsao.index, df_previsao['previsão']):
    plt.text(x, y, f'{y:.2f}', fontsize=9, ha='right')

plt.plot(df_meta.index, df_meta['meta'], label='Meta 2024', linestyle=':', color='red', marker='x')
for x, y in zip(df_meta.index, df_meta['meta']):
    plt.text(x, y, f'{y:.2f}', fontsize=9, ha='right')

plt.title('Resultados e Previsões com Suavização Tripla Aditiva e Metas')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.grid(True)
plt.show()

# Exportar para CSV
df_complete = pd.concat([df, df_previsao], axis=0)
df_complete['meta'] = df_meta['meta']
df_complete.to_csv('resultados_previsoes.csv', decimal=',')
