# Previsão de Evasão de Funcionários da IBM

*Descrição*
Este projeto tem como objetivo prever a probabilidade de um funcionário se demitir da IBM. Utilizamos um conjunto de dados fornecido pela empresa, que inclui informações como idade, departamento, frequência de viagens, salário, nível de satisfação no trabalho e tempo de serviço.

Utilizamos o Python como linguagem de programação e os modelos Logistic Regression, Random Forest e XGBoost para prever a evasão de funcionários. Utilizamos as métricas Precision e Recall e a Matriz de Confusão para avaliar a precisão do modelo.

*Hipóteses que tentaremos responder:*

Existe alguma relação entre a idade do funcionário e sua probabilidade de evasão?
Funcionários da área de vendas têm maior probabilidade de evasão do que outros departamentos?
Funcionários que viajam com mais frequência têm maior probabilidade de evasão?
Qual é a influência do salário na probabilidade de evasão?
Existe alguma relação entre o nível de satisfação com o trabalho e a probabilidade de evasão?
Qual é a influência do tempo que um funcionário passou na empresa na probabilidade de evasão?
Funcionários que receberam uma promoção recentemente têm menor probabilidade de evasão?
Conjunto de Dados
O conjunto de dados que utilizamos foi fornecido pela IBM e inclui informações sobre os funcionários da empresa. O conjunto de dados possui X amostras e Y características. Ele está localizado no arquivo "IBM_Employee.csv" neste repositório.

Arquivos do Projeto
O projeto possui os seguintes arquivos:

IBM Hr Analytics.csv: conjunto de dados utilizado no projeto.
IBM.ipynb: um notebook Jupyter que contém o código utilizado para análise e previsão.
README.md: este arquivo.
Resultados
Os resultados da previsão foram avaliados com as métricas de Precision e Recall e a Matriz de Confusão. Os resultados são descritos no arquivo IBM Hr Analytics.ipynb.

*Conclusão*
Com base nos resultados, concluímos que o modelo de Regressão Logística foi o que apresentou a melhor capacidade de prever a probabilidade de evasão dos funcionários da IBM.

Referências
Dataset: IBM HR Analytics Employee Attrition & Performance
Bibliotecas utilizadas: pandas, matplotlib, seaborn, sklearn.
