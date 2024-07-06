import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Dados fornecidos
meta_2024 = {
    'MES 2024': ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez'],
    'IPQ': [21.28, 42.55, 8.51, 46.81, 25.53, 34.04, 34.04, 55.32, 29.79, 25.53, 38.30, 38.30],
    'IMV': [2.00, 0.57, 3.15, 1.14, 1.14, 1.72, 0.57, 2.86, 1.43, 2.57, 0.86, 1.14]
}

df_meta = pd.DataFrame(meta_2024)

# Dados dos resultados para o 19 BPM
resultados_19BPM = {
    'mês': ['jan/24', 'fev/24', 'mar/24', 'abr/24', 'mai/24', 'jun/24'],
    'imv': [1.74, 0.29, 2.03, 3.19, 3.78, 3.19],
    'ipq': [106.38, 12.77, 29.79, 25.53, 38.3, 38.3]
}

df_resultados = pd.DataFrame(resultados_19BPM)

# Ajustar um polinômio cúbico aos dados
X = np.arange(1, len(df_meta) + 1)
y = df_meta['IMV']
coefs = np.polyfit(X, y, 3)
poly = np.poly1d(coefs)
X_smooth = np.linspace(X.min(), X.max(), 300)
y_smooth = poly(X_smooth)

# Calcular o desvio padrão dos resíduos
residuos = y - poly(X)
std_dev = np.std(residuos)

# Limites superior e inferior baseados no desvio padrão
limite_superior = y_smooth + std_dev
limite_inferior = y_smooth - std_dev

# Plotar os dados
plt.figure(figsize=(14, 7))
sns.set(style="whitegrid")

# Plotar os valores reais
plt.scatter(X, df_meta['IMV'], color='blue', label='IMV Meta 2024')
plt.scatter(np.arange(1, len(df_resultados) + 1), df_resultados['imv'], color='red', label='IMV 19 BPM 2024')

# Plotar a curva ajustada
plt.plot(X_smooth, y_smooth, color='blue', linestyle='-', linewidth=2, label='Curva Polinomial Cúbica')

# Plotar os limites superior e inferior
plt.plot(X_smooth, limite_superior, color='gray', linestyle='--', linewidth=1)
plt.plot(X_smooth, limite_inferior, color='gray', linestyle='--', linewidth=1)

# Preencher a área entre os limites
plt.fill_between(X_smooth, limite_inferior, limite_superior, color='gray', alpha=0.8, label='Limite Superior/Inferior')

# Adicionar rótulos e título
plt.title('IMV e IPQ - Meta 2024 vs Resultados 19 BPM')
plt.xlabel('Mês')
plt.ylabel('IMV')
plt.xticks(X, df_meta['MES 2024'])
plt.legend()

# Anotação dos valores reais
for i in range(len(df_resultados)):
    plt.text(i+1, df_resultados['imv'][i] + 0.1, f"{df_resultados['imv'][i]:.2f}", ha='center')

# Fonte e anotação
plt.annotate('Fonte: CGA/DOP - Base BISP auditada. Extração com dados até 01/05/2024',
             xy=(0.5, -0.1), xycoords='axes fraction', ha='center', fontsize=10)

# Exibir o gráfico
plt.show()

# Exibir a função gerada
print(f"Função gerada: {poly}")
