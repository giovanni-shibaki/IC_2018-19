# IC_2018-19 - Inteligência Artifical e Aplicações

Iniciação Científica realizada durante 1 ano a partir do segundo semestre de 2018.


# Objetivos

Implementação de um algoritmo de IA a fim de mensurar o índice de evasão dos clientes de uma companhia de Telecom.

## Etapas de Desenvolvimento

**1 - Implementação do Banco de Dados**
	Para mensurar o índice de evasão foi desenvolvido um banco de dados com o histórico de transações dos clientes da companhia e implementado diferentes algoritmos de IA para prever a taxa de cancelamento dos mesmos na rede de Telecom

**2 - Modelagem dos Dados**

Para armazenamento e modelação dos dados da companhia de telecom, foi utilizado um banco de dados MySQL.

Após a indexação dos números de contrato de cada cliente da companhia, os dados foram separados de acordo com as ações de cada usuário. Assim, a frequência acumulada do número de ações de cada tipo (taxa de recarga, uso de dados móveis, ligações efetuadas, etc) foi separada semanalmente no prazo de um mês, resultando em 16 características nas quais os classificadores em linguagem Python foram implementados.

**3 - Programação**
Utilizando-se da biblioteca de Aprendizado de Máquina Scikit-Learn foi desenvolvido um programa em linguagem Python que recebia os dados já modelados e por meio de um script genérico para os diferentes classificadores, realizava as fases de treino e de teste e por fim classificava cada usuário como ATIVO ou CANCELADO e exibia a probabilidade de um usuário específico, no prazo de um mês, de cancelar sua linha.




# Classificadores observados

Durante o desenvolvimento do projeto foi pesquisado o funcionamento e aplicação básica de cada um dos 10 algoritmos de classificação, dentre eles:

- Gradient Boosting
- Random Forest
- Regressão Logística
- K-Nearest Neighbor

# Arquivos

Neste repositório estão contidos alguns dos programas desenvolvidos durante a realização do projeto, os resultados obtidos e também a documentação feita para esta Iniciação Científica.
