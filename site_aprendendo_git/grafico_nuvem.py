import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Simulando dados para o exemplo
np.random.seed(42)
data = {
    'IMV': np.random.uniform(-15, 15, 100),
    'IPQ': np.random.uniform(0, 700, 100),
    'Excesso de crimes sobre a meta': np.random.uniform(-15, 15, 100),
    'Cia': np.random.choice(['34 CIA PM IND', '44 BPM', '70 BPM', '90 BPM'], 100)
}

df = pd.DataFrame(data)

# Criando o gráfico
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")

# Criar o scatterplot com coloração
scatter = sns.scatterplot(x='IPQ', y='IMV', hue='Excesso de crimes sobre a meta', data=df, palette='RdYlGn', edgecolor='w', s=100)

# Ajustar os limites dos eixos
scatter.set_xlim(0, 700)
scatter.set_ylim(-15, 15)

# Adicionar linha de tendência
sns.regplot(x='IPQ', y='IMV', data=df, scatter=False, color='blue', lowess=True)

# Adicionar rótulos e título
plt.title('Correlação: -0.29')
plt.xlabel('IPQ')
plt.ylabel('IMV')

# Adicionar retângulos coloridos estilo ninebox
regions = [
    (0, 233, -5, 5, 'red', '4 (Baixo esforço)'),
    (0, 233, 5, 15, 'orange', '1 (Enigma resultado)'),
    (0, 233, -15, -5, 'red', '7 (Insuficiente)'),
    (233, 466, -5, 5, 'yellow', '5 (Médio esforço e resultado)'),
    (233, 466, 5, 15, 'green', '2 (Forte resultado)'),
    (233, 466, -15, -5, 'red', '8 (Baixo Resultado)'),
    (466, 700, -5, 5, 'green', '6 (Forte esforço)'),
    (466, 700, 5, 15, 'green', '3 (Alto desempenho)'),
    (466, 700, -15, -5, 'orange', '9 (Enigma esforço)')
]

for x_min, x_max, y_min, y_max, color, label in regions:
    plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, edgecolor=color, facecolor=color, alpha=0.1, lw=2))
    plt.text((x_min + x_max) / 2, (y_min + y_max) / 2, label, fontsize=12, ha='center', color='black')

# Ajustar legenda
plt.legend(title='Excesso de crimes sobre a meta', bbox_to_anchor=(1.05, 1), loc='upper left')

# Fonte e anotação
plt.annotate('Fonte: CGA/DOP - Base BISP auditada. Extração com dados até 01/05/2024',
             xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontsize=10)

# Exibir o gráfico
plt.show()
