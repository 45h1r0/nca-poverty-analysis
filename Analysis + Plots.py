"""
Análise Comparativa da Cesta de Consumo em Pernambuco (POF 2018)
Análise 1: Cesta Bruta (sem refinamento)
Análise 2: Cesta Refinada (com base nos critérios de Mancini)
"""
#Bibliotecas necessárias:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -- ETAPA 1: Carregamento e Preparação Inicial --
try:
    df_raw = pd.read_csv('Consume_Basket_DRP/POF2018.csv')
except FileNotFoundError:
    print("Arquivo 'aula bem estar -pof 2018.csv' não encontrado.")
    exit()

# Filtrando os dados para Pernambuco (UF = 26)
df_pe_raw = df_raw[df_raw['uf'] == 26].copy()

# -- Função de Classificação (será usada em ambas as análises) --
def get_correct_group_name(code):
    prefix = code // 100000
    if 11 <= prefix <= 12: return 'Alimentação'
    if 21 <= prefix <= 23: return 'Habitação'
    if 31 <= prefix <= 33: return 'Habitação'
    if 41 <= prefix <= 44: return 'Vestuário'
    if prefix == 51: return 'Transporte'
    if 61 <= prefix <= 63: return 'Saúde e Cuidados Pessoais'
    if prefix == 71: return 'Despesas Pessoais'
    if prefix == 72: return 'Educação, Lazer e Cultura'
    if prefix == 81: return 'Educação, Lazer e Cultura'
    if prefix == 91: return 'Outras Despesas'
    else: return 'Outras Despesas'

#==============================================================================
# ANÁLISE 1: CESTA BRUTA (SEM REFINAMENTO)
#==============================================================================
print("\n" + "#"*70)
print("# ANÁLISE 1: CESTA DE CONSUMO BRUTA (SEM REFINAMENTO)")
print("#"*70)

# Criando a classificação de grupos
df_pe_raw['nome_grupo'] = df_pe_raw['cod_subitem'].apply(get_correct_group_name)

# Agregação dos gastos
df_agregado_bruto = pd.pivot_table(
    df_pe_raw,
    values='gasto',
    index=['domicilio', 'uf'],
    columns=['nome_grupo'],
    aggfunc='sum',
    fill_value=0
)

# Cálculo do gasto nominal
colunas_de_gasto_bruto = [col for col in df_agregado_bruto.columns]
df_agregado_bruto['gasto_nominal'] = df_agregado_bruto[colunas_de_gasto_bruto].sum(axis=1)

# Deflação, Outliers e Análise de Sensibilidade
fator_deflacao = 0.94  # ### ALTERAÇÃO: Fator de deflação ajustado para 1.01 ###
df_agregado_bruto['gasto_real'] = df_agregado_bruto['gasto_nominal'] / fator_deflacao
df_agregado_bruto['log_gasto_real'] = np.log(df_agregado_bruto['gasto_real'] + 1)
median_log_gasto_bruto = df_agregado_bruto['log_gasto_real'].median()
mad_bruto = np.median(np.abs(df_agregado_bruto['log_gasto_real'] - median_log_gasto_bruto))
sigma_mad_bruto = 1.4826 * mad_bruto
limite_superior_bruto = median_log_gasto_bruto + 3 * sigma_mad_bruto
limite_inferior_bruto = median_log_gasto_bruto - 3 * sigma_mad_bruto
df_final_bruto = df_agregado_bruto[
    (df_agregado_bruto['log_gasto_real'] >= limite_inferior_bruto) &
    (df_agregado_bruto['log_gasto_real'] <= limite_superior_bruto)
].copy()

print(f"\nAnálise (Bruta) final será feita com {len(df_final_bruto)} domicílios.")
linha_pobreza_bruta = df_final_bruto['gasto_real'].quantile(0.35)
print(f"Linha de pobreza (Bruta): R$ {linha_pobreza_bruta:.2f}")

# ### APRESENTAÇÃO DOS RESULTADOS DA CESTA BRUTA ###
# --- PARTE 1: Tabela Resumo com Grupos e Pesos Gerais (Bruta) ---
print("\n" + "="*50)
print(" PARTE 1: Tabela Resumo da Cesta de Consumo (Bruta)")
print("="*50)
resumo_data_bruto = []
gasto_total_bruto = df_final_bruto[colunas_de_gasto_bruto].sum().sum()
pesos_bruto = {grupo: df_final_bruto[grupo].sum() / gasto_total_bruto for grupo in colunas_de_gasto_bruto}
grupos_ordenados_bruto = sorted(pesos_bruto.items(), key=lambda item: item[1], reverse=True)
for nome_grupo, peso in grupos_ordenados_bruto:
    resumo_data_bruto.append([nome_grupo, f"{peso:.2%}"])
df_resumo_bruto = pd.DataFrame(resumo_data_bruto, columns=['Grupo de Consumo', 'Peso Relativo no Gasto Total'])
print(df_resumo_bruto.to_string(index=False))

# --- PARTE 2: Tabelas de cada grupo, separadamente (Bruta) ---
print("\n\n" + "="*60)
print(" PARTE 2: Detalhamento de Itens por Grupo (Cesta Bruta)")
print("="*60)
pd.set_option('display.max_rows', None)
for nome_grupo, peso in grupos_ordenados_bruto:
    print(f"\n\n--- Grupo: {nome_grupo} ---")
    
    # ### ALTERAÇÃO: Cálculo do peso do subitem dentro do grupo ###
    grupo_df_bruto = df_pe_raw[df_pe_raw['nome_grupo'] == nome_grupo]
    gasto_total_grupo_bruto = grupo_df_bruto['gasto'].sum()
    
    # Agrupa por subitem para somar os gastos
    subitem_gastos_bruto = grupo_df_bruto.groupby(['cod_subitem', 'subitem'])['gasto'].sum().reset_index()
    # Calcula o peso de cada subitem
    subitem_gastos_bruto['Peso no Grupo (%)'] = (subitem_gastos_bruto['gasto'] / gasto_total_grupo_bruto) * 100
    subitem_gastos_bruto = subitem_gastos_bruto.sort_values(by='Peso no Grupo (%)', ascending=False)
    subitem_gastos_bruto['Peso no Grupo (%)'] = subitem_gastos_bruto['Peso no Grupo (%)'].map('{:.2f}%'.format)
    
    print(subitem_gastos_bruto[['cod_subitem', 'subitem', 'Peso no Grupo (%)']].to_string(index=False))
pd.reset_option('display.max_rows')


#==============================================================================
# ANÁLISE 2: CESTA REFINADA (COM BASE EM MANCINI)
#==============================================================================
print("\n\n" + "#"*70)
print("# ANÁLISE 2: CESTA DE CONSUMO REFINADA (PÓS-AJUSTES)")
print("#"*70)

# Refinamento da Cesta de Consumo
print("\nRefinando a cesta de consumo com base nos critérios de Mancini...")
itens_para_excluir = [
    1201009, 3101002, 3101003, 3101015, 3101016, 3101017, 3201001, 3201002, 3201006, 3201013, 3201021, 3201050, 3202001, 3202003, 3202028,
    2103039, 2103040, 2103042, 2103048, 2103055, 6202004, 5102053, 7201010, 7201019, 7201095, 1201048, 1201061, 1201003, 1201001, 1201007,
    7101034, 7101036, 7201063, 7201090, 2103005, 2103049, 3301022, 5102004, 5102011, 5102051, 5101010, 5102037, 5102010, 4301002, 4301004,
    3102007, 3102009, 3102010

]
df_pe_refinado = df_pe_raw[~df_pe_raw['cod_subitem'].isin(itens_para_excluir)].copy()
print(f"{len(df_pe_raw) - len(df_pe_refinado)} registros de despesas foram removidos da análise.")

# Agregação dos gastos com o DataFrame refinado
df_agregado_refinado = pd.pivot_table(
    df_pe_refinado,
    values='gasto',
    index=['domicilio', 'uf'],
    columns=['nome_grupo'],
    aggfunc='sum',
    fill_value=0
)

# Cálculo do gasto nominal
colunas_de_gasto_refinado = [col for col in df_agregado_refinado.columns]
df_agregado_refinado['gasto_nominal'] = df_agregado_refinado[colunas_de_gasto_refinado].sum(axis=1)

# Deflação, Outliers e Análise de Sensibilidade
fator_deflacao = 0.94 # ### ALTERAÇÃO: Fator de deflação ajustado para 1.01 ###
df_agregado_refinado['gasto_real'] = df_agregado_refinado['gasto_nominal'] / fator_deflacao
df_agregado_refinado['log_gasto_real'] = np.log(df_agregado_refinado['gasto_real'] + 1)
median_log_gasto_refinado = df_agregado_refinado['log_gasto_real'].median()
mad_refinado = np.median(np.abs(df_agregado_refinado['log_gasto_real'] - median_log_gasto_refinado))
sigma_mad_refinado = 1.4826 * mad_refinado
limite_superior_refinado = median_log_gasto_refinado + 3 * sigma_mad_refinado
limite_inferior_refinado = median_log_gasto_refinado - 3 * sigma_mad_refinado
df_final_refinado = df_agregado_refinado[
    (df_agregado_refinado['log_gasto_real'] >= limite_inferior_refinado) &
    (df_agregado_refinado['log_gasto_real'] <= limite_superior_refinado)
].copy()

print(f"\nAnálise (Refinada) final será feita com {len(df_final_refinado)} domicílios.")
linha_pobreza_refinada = df_final_refinado['gasto_real'].quantile(0.35)
print(f"Linha de pobreza (Refinada): R$ {linha_pobreza_refinada:.2f}")

# ### APRESENTAÇÃO DOS RESULTADOS DA CESTA REFINADA ###
# --- PARTE 1: Tabela Resumo com Grupos e Pesos Gerais (Refinada) ---
print("\n" + "="*50)
print(" PARTE 1: Tabela Resumo da Cesta de Consumo (Refinada)")
print("="*50)
resumo_data_refinado = []
gasto_total_refinado = df_final_refinado[colunas_de_gasto_refinado].sum().sum()
pesos_refinado = {grupo: df_final_refinado[grupo].sum() / gasto_total_refinado for grupo in colunas_de_gasto_refinado}
grupos_ordenados_refinado = sorted(pesos_refinado.items(), key=lambda item: item[1], reverse=True)
for nome_grupo, peso in grupos_ordenados_refinado:
    resumo_data_refinado.append([nome_grupo, f"{peso:.2%}"])
df_resumo_refinado = pd.DataFrame(resumo_data_refinado, columns=['Grupo de Consumo', 'Peso Relativo no Gasto Total'])
print(df_resumo_refinado.to_string(index=False))

# --- PARTE 2: Tabelas de cada grupo, separadamente (Refinada) ---
print("\n\n" + "="*60)
print(" PARTE 2: Detalhamento de Itens por Grupo (Cesta Refinada)")
print("="*60)
pd.set_option('display.max_rows', None)
for nome_grupo, peso in grupos_ordenados_refinado:
    print(f"\n\n--- Grupo: {nome_grupo} ---")
    
    # ### ALTERAÇÃO: Cálculo do peso do subitem dentro do grupo refinado ###
    grupo_df_refinado = df_pe_refinado[df_pe_refinado['nome_grupo'] == nome_grupo]
    gasto_total_grupo_refinado = grupo_df_refinado['gasto'].sum()

    subitem_gastos_refinado = grupo_df_refinado.groupby(['cod_subitem', 'subitem'])['gasto'].sum().reset_index()
    subitem_gastos_refinado['Peso no Grupo (%)'] = (subitem_gastos_refinado['gasto'] / gasto_total_grupo_refinado) * 100
    subitem_gastos_refinado = subitem_gastos_refinado.sort_values(by='Peso no Grupo (%)', ascending=False)
    subitem_gastos_refinado['Peso no Grupo (%)'] = subitem_gastos_refinado['Peso no Grupo (%)'].map('{:.2f}%'.format)

    print(subitem_gastos_refinado[['cod_subitem', 'subitem', 'Peso no Grupo (%)']].to_string(index=False))
pd.reset_option('display.max_rows')


#==============================================================================
# ETAPA DE GRÁFICOS COMPARATIVOS
#==============================================================================
print("\n\n" + "#"*70)
print("# ETAPA DE GRÁFICOS: GERANDO GRÁFICOS COMPARATIVOS")
print("#"*70)

# Gráfico 1: Comparação dos Pesos da Cesta
df_resumo_bruto_plot = df_resumo_bruto.set_index('Grupo de Consumo')
df_resumo_bruto_plot.columns = ['Peso (Bruto)']
df_resumo_bruto_plot['Peso (Bruto)'] = df_resumo_bruto_plot['Peso (Bruto)'].str.replace('%', '').astype(float)
df_resumo_refinado_plot = df_resumo_refinado.set_index('Grupo de Consumo')
df_resumo_refinado_plot.columns = ['Peso (Refinado)']
df_resumo_refinado_plot['Peso (Refinado)'] = df_resumo_refinado_plot['Peso (Refinado)'].str.replace('%', '').astype(float)
df_comparativo = pd.concat([df_resumo_bruto_plot, df_resumo_refinado_plot], axis=1).sort_values(by='Peso (Refinado)')
fig_comp, ax_comp = plt.subplots(figsize=(12, 8))
df_comparativo.plot(kind='barh', ax=ax_comp)
ax_comp.set_title('Comparação dos Pesos na Cesta de Consumo (Bruta vs. Refinada)')
ax_comp.set_xlabel('Peso Relativo (%)')
plt.savefig('grafico_comparativo_pesos.png', bbox_inches='tight', dpi=300)
plt.close(fig_comp)
print("Gráfico comparativo dos pesos salvo como 'grafico_comparativo_pesos.png'")

# Gráfico 2 - Comparação das Distribuições de Gasto
fig_dist_comp, ax_dist_comp = plt.subplots(figsize=(12, 7))
sns.kdeplot(df_final_bruto['gasto_real'], ax=ax_dist_comp, label='Distribuição Bruta', fill=True)
sns.kdeplot(df_final_refinado['gasto_real'], ax=ax_dist_comp, label='Distribuição Refinada', fill=True)
ax_dist_comp.axvline(linha_pobreza_bruta, color='blue', linestyle='--', label=f'Linha Pobreza Bruta (R${linha_pobreza_bruta:.2f})')
ax_dist_comp.axvline(linha_pobreza_refinada, color='red', linestyle='--', label=f'Linha Pobreza Refinada (R${linha_pobreza_refinada:.2f})')
ax_dist_comp.set_title('Comparação das Distribuições de Gasto Real')
ax_dist_comp.set_xlabel('Gasto Real por Domicílio (R$)')
ax_dist_comp.legend()
plt.savefig('grafico_comparativo_distribuicoes.png', bbox_inches='tight', dpi=300)
plt.close(fig_dist_comp)
print("Gráfico comparativo das distribuições salvo como 'grafico_comparativo_distribuicoes.png'")
#==============================================================================  
# NOVA TABELA: COMPARAÇÃO EM TORNO DA LINHA DE POBREZA (±20%)  
#==============================================================================  
print("\n" + "="*80)
print("TABELA COMPARATIVA: REAL VS NOMINAL (±10% DA LINHA DE POBREZA)")
print("="*80)

linha_pobreza = linha_pobreza_refinada

percentuais = [0.10, 0.05, 0.0, -0.05, -0.10]
linhas = []
for pct in percentuais:
    linha_ajustada = linha_pobreza * (1 + pct)
    hcr_adj = (df_final_refinado['gasto_real'] <= linha_ajustada).mean() * 100  # %
    hcr = (df_final_refinado['gasto_nominal'] <= linha_ajustada).mean() * 100   # %
    change = hcr_adj - hcr
    if pct == 0:
        label = "P35"
    else:
        label = f"{int(pct*100):+d}%"
    linhas.append([label, f"{linha_ajustada:.2f}", f"{hcr_adj:.1f}", f"{hcr:.1f}", f"{change:.1f}"])

df_metodologia = pd.DataFrame(linhas, columns=[
    "Linha de pobreza", "BRL", "HCR_adj (%)", "HCR (%)", "Change from HCR (%)"
])

print(df_metodologia.to_string(index=False))

#Gráfico 3: Distribuição Cumulativa Real vs Nominal 
# ============================================================================== 
print("\n\n" + "#"*70)  
print("# NOVA ETAPA: DISTRIBUIÇÃO CUMULATIVA DA CESTA REFINADA")  
print("#"*70)  

# Dados de base  
gasto_real = df_final_refinado['gasto_real'].sort_values().reset_index(drop=True)  
gasto_nominal = df_final_refinado['gasto_nominal'].sort_values().reset_index(drop=True)  
n = len(gasto_real)  
proporcao = np.arange(1, n+1) / n  

# Linha de pobreza (P35)  
prop_pobreza = (gasto_real <= linha_pobreza).mean()

# Faixa de ±20%  
limite_inferior = linha_pobreza * 0.8  
limite_superior = linha_pobreza * 1.2  

# Definindo os 11 valores analisados igualmente espaçados na faixa ±20% da linha de pobreza
valores_analisados = np.linspace(limite_inferior, limite_superior, 11)

# Gerando gráfico  
fig_cesta_refinada, ax_ref = plt.subplots(figsize=(12, 7))
ax_ref.plot(gasto_real, proporcao, label='Gasto Real (Deflacionado)', color='green')
ax_ref.plot(gasto_nominal, proporcao, label='Gasto Nominal', color='orange')

# Linha pontilhada vermelha passando pelo ponto da linha de pobreza
ax_ref.axvline(linha_pobreza, color='red', linestyle='--', linewidth=2, label=f'Linha de Pobreza (R$ {linha_pobreza:.2f})')
ax_ref.axvspan(limite_inferior, limite_superior, color='gray', alpha=0.2, label='±20% em torno da Pobreza')

# Marcadores e rótulos nos 11 pontos da tabela
faixa_linhas = []
for valor in valores_analisados:
    prop_real = (gasto_real <= valor).mean()
    prop_nominal = (gasto_nominal <= valor).mean()

    # Marcador e rótulo para gasto_real
    ax_ref.plot(valor, prop_real, 'o', color='green')
    ax_ref.text(valor, prop_real + 0.01, f'R${valor:.2f}', color='green', fontsize=8, ha='center', va='bottom')
    # Marcador e rótulo para gasto_nominal
    ax_ref.plot(valor, prop_nominal, 's', color='orange')
    ax_ref.text(valor, prop_nominal - 0.01, f'R${valor:.2f}', color='orange', fontsize=8, ha='center', va='top')

    faixa_linhas.append([
        f'R${valor:.2f}',
        f'{prop_real*100:.2f}',
        f'{prop_nominal*100:.2f}',
        f'{(prop_real-prop_nominal)*100:.2f}'
    ])

df_comparacao_faixa = pd.DataFrame(
    faixa_linhas,
    columns=['Valor (R$)', 'Proporção Real (%)', 'Proporção Nominal (%)', 'Diferença Real-Nominal (%)']
)

ax_ref.set_xlim(limite_inferior, limite_superior)
ax_ref.set_ylim((gasto_real <= limite_inferior).mean(), (gasto_real <= limite_superior).mean())
ax_ref.set_xlabel('Gasto por Domicílio (R$)')
ax_ref.set_ylabel('Proporção acumulada da população')
ax_ref.set_title('Distribuição Cumulativa - Cesta Refinada\n(11 pontos da faixa ±20% da linha de pobreza)')
ax_ref.legend()
ax_ref.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('grafico_cumulativo_cesta_refinada.png', bbox_inches='tight', dpi=300)
plt.close(fig_cesta_refinada)
print("Gráfico cumulativo da cesta refinada salvo como 'grafico_cumulativo_cesta_refinada.png'")


# Exportando para o Excel na aba nova  
try:  
    with pd.ExcelWriter('relatorio_cesta_de_consumo.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:  
        df_comparacao_faixa.to_excel(writer, sheet_name='Comparacao_Faixa_±20%', index=False)  
    print("Nova aba 'Comparacao_Faixa_±20%' adicionada ao arquivo Excel com sucesso!")  
except Exception as e:  
    print(f"Erro ao adicionar nova aba ao Excel: {e}")

# ### ETAPA EXTRA: EXPORTAR TABELAS PARA ARQUIVO EXCEL ###
print("\n\n" + "#"*70)
print("# ETAPA EXTRA: EXPORTANDO RESULTADOS PARA EXCEL")
print("#"*70)

try:
    with pd.ExcelWriter('relatorio_cesta_de_consumo.xlsx', engine='openpyxl') as writer:
        # Escrevendo os resumos
        df_resumo_bruto.to_excel(writer, sheet_name='Resumo_Cesta_Bruta', index=False)
        df_resumo_refinado.to_excel(writer, sheet_name='Resumo_Cesta_Refinada', index=False)

        # ### ALTERAÇÃO: Incluindo os pesos dos subitens no Excel ###
        # Preparando e escrevendo os detalhes da Cesta Bruta com pesos
        itens_detalhados_brutos = []
        for nome_grupo, peso in grupos_ordenados_bruto:
            grupo_df_bruto = df_pe_raw[df_pe_raw['nome_grupo'] == nome_grupo]
            gasto_total_grupo_bruto = grupo_df_bruto['gasto'].sum()
            subitem_gastos_bruto = grupo_df_bruto.groupby(['cod_subitem', 'subitem'])['gasto'].sum().reset_index()
            subitem_gastos_bruto['Peso no Grupo (%)'] = (subitem_gastos_bruto['gasto'] / gasto_total_grupo_bruto) * 100
            subitem_gastos_bruto = subitem_gastos_bruto.sort_values(by='Peso no Grupo (%)', ascending=False)
            subitem_gastos_bruto['grupo'] = nome_grupo
            itens_detalhados_brutos.append(subitem_gastos_bruto)
        df_detalhes_bruto = pd.concat(itens_detalhados_brutos, ignore_index=True)
        df_detalhes_bruto = df_detalhes_bruto[['grupo', 'cod_subitem', 'subitem', 'gasto', 'Peso no Grupo (%)']]
        df_detalhes_bruto.to_excel(writer, sheet_name='Itens_Cesta_Bruta', index=False)

        # Preparando e escrevendo os detalhes da Cesta Refinada com pesos
        itens_detalhados_refinados = []
        for nome_grupo, peso in grupos_ordenados_refinado:
            grupo_df_refinado = df_pe_refinado[df_pe_refinado['nome_grupo'] == nome_grupo]
            gasto_total_grupo_refinado = grupo_df_refinado['gasto'].sum()
            subitem_gastos_refinado = grupo_df_refinado.groupby(['cod_subitem', 'subitem'])['gasto'].sum().reset_index()
            subitem_gastos_refinado['Peso no Grupo (%)'] = (subitem_gastos_refinado['gasto'] / gasto_total_grupo_refinado) * 100
            subitem_gastos_refinado = subitem_gastos_refinado.sort_values(by='Peso no Grupo (%)', ascending=False)
            subitem_gastos_refinado['grupo'] = nome_grupo
            itens_detalhados_refinados.append(subitem_gastos_refinado)
        df_detalhes_refinado = pd.concat(itens_detalhados_refinados, ignore_index=True)
        df_detalhes_refinado = df_detalhes_refinado[['grupo', 'cod_subitem', 'subitem', 'gasto', 'Peso no Grupo (%)']]
        df_detalhes_refinado.to_excel(writer, sheet_name='Itens_Cesta_Refinada', index=False)
        
        # NOVO: Adicionando a tabela de análise Real vs Nominal
        df_metodologia.to_excel(writer, sheet_name='Analise_Real_vs_Nominal', index=False)

    print("Arquivo 'relatorio_cesta_de_consumo.xlsx' gerado com sucesso!")
    print("O arquivo contém 5 abas incluindo a nova análise Real vs Nominal.")

except ImportError:
    print("\nAVISO: Para salvar em Excel, a biblioteca 'openpyxl' é necessária.")
    print("Por favor, instale-a usando o comando: pip install openpyxl")
except Exception as e:
    print(f"\nOcorreu um erro ao salvar o arquivo Excel: {e}")

print("\n" + "#"*70)
print("# ANÁLISE COMPLETA FINALIZADA")
print("#"*70)
print("Gráficos gerados:")
print("1. grafico_comparativo_pesos.png")
print("2. grafico_comparativo_distribuicoes.png")
print("3. grafico_distribuicao_cumulativa_real_vs_nominal.png (NOVO)")
print("\nArquivos Excel:")
print("1. relatorio_cesta_de_consumo.xlsx (atualizado com análise Real vs Nominal)")
print("\n" + "="*70)
print("RESUMO DA ANÁLISE CUMULATIVA:")
print("="*70)
print("O gráfico mostra a distribuição cumulativa comparando:")
print("- Gasto Real (deflacionado) vs Gasto Nominal")
print("- Para ambas as cestas (Bruta e Refinada)")
print("- Com destaque especial para a linha de pobreza (P35) em vermelho pontilhado")
print("- Marcadores nos percentis: P15, P20, P25, P30, P40, P45, P50, P55")
print("- Valores monetários exibidos para cada ponto de interesse")

# === NOVA TABELA: COMPOSIÇÃO DA CESTA ATÉ A LINHA DE POBREZA (P35) ===
print("\n" + "="*80)
print("COMPOSIÇÃO DA CESTA ATÉ A LINHA DE POBREZA (P35)")
print("="*80)

# Filtra domicílios até a linha de pobreza
df_ate_pobreza = df_final_refinado[df_final_refinado['gasto_real'] <= linha_pobreza_refinada].copy()

# Grupos de consumo (exclui colunas técnicas)
grupos = [col for col in df_ate_pobreza.columns if col not in ['gasto_nominal', 'gasto_real', 'log_gasto_real', 'uf']]

# Gasto médio por grupo
gasto_medio_grupo = df_ate_pobreza[grupos].mean()
gasto_total_medio = gasto_medio_grupo.sum()
resumo_grupo = []
for grupo in grupos:
    peso_grupo = gasto_medio_grupo[grupo] / gasto_total_medio if gasto_total_medio > 0 else 0
    resumo_grupo.append([grupo, gasto_medio_grupo[grupo], peso_grupo])

df_composicao_grupo = pd.DataFrame(resumo_grupo, columns=[
    'Grupo', 'Gasto Médio no Grupo (R$)', 'Peso no Total (%)'
])
df_composicao_grupo['Peso no Total (%)'] = df_composicao_grupo['Peso no Total (%)'].map('{:.2%}'.format)

# Gasto médio por subitem dentro de cada grupo
resumo_subitem = []
for grupo in grupos:
    codigos = df_pe_refinado[df_pe_refinado['nome_grupo'] == grupo]['cod_subitem'].unique()
    for cod in codigos:
        nome_subitem = df_pe_refinado[df_pe_refinado['cod_subitem'] == cod]['subitem'].iloc[0]
        # Soma o gasto desse subitem para domicílios até a linha de pobreza
        gasto_subitem = df_pe_refinado[
            (df_pe_refinado['cod_subitem'] == cod) &
            (df_pe_refinado['domicilio'].isin(df_ate_pobreza.index.get_level_values('domicilio')))
        ]['gasto'].mean()
        peso_subitem_grupo = gasto_subitem / gasto_medio_grupo[grupo] if gasto_medio_grupo[grupo] > 0 else 0
        peso_subitem_total = gasto_subitem / gasto_total_medio if gasto_total_medio > 0 else 0
        resumo_subitem.append([
            grupo, cod, nome_subitem, gasto_subitem, peso_subitem_grupo, peso_subitem_total
        ])

df_composicao_subitem = pd.DataFrame(resumo_subitem, columns=[
    'Grupo', 'Código Subitem', 'Nome Subitem', 'Gasto Médio Subitem (R$)', 'Peso no Grupo (%)', 'Peso no Total (%)'
])
df_composicao_subitem['Peso no Grupo (%)'] = df_composicao_subitem['Peso no Grupo (%)'].map('{:.2%}'.format)
df_composicao_subitem['Peso no Total (%)'] = df_composicao_subitem['Peso no Total (%)'].map('{:.2%}'.format)

print("\nResumo por Grupo:")
print(df_composicao_grupo.to_string(index=False))
print("\nResumo por Subitem:")
print(df_composicao_subitem.to_string(index=False))

# Exporta para Excel
try:
    with pd.ExcelWriter('relatorio_cesta_de_consumo.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_composicao_grupo.to_excel(writer, sheet_name='Cesta_ate_Linha_Pobreza_Grupo', index=False)
        df_composicao_subitem.to_excel(writer, sheet_name='Cesta_ate_Linha_Pobreza_Subitem', index=False)
    print("Tabelas de composição da cesta até a linha de pobreza adicionadas ao Excel!")
except Exception as e:
    print(f"Erro ao exportar composição da cesta até a linha de pobreza: {e}")