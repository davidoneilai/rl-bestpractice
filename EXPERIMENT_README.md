# 🚀 Experimentos de Reinforcement Learning - Métricas TensorBoard

Este projeto implementa um sistema completo de monitoramento e análise para experimentos de Reinforcement Learning usando DQN e A3C no ambiente LunarLander-v3.

## 📊 Métricas Implementadas

### DQN (Deep Q-Network)
- **Loss da Rede Neural**: Perda MSE entre Q-values preditos e target
- **Recompensa por Episódio**: Performance do agente
- **Duração do Episódio**: Número de passos até terminar
- **Epsilon**: Valor atual da exploração epsilon-greedy
- **Recompensa Média (100 episódios)**: Tendência de performance

### A3C (Asynchronous Advantage Actor-Critic)
- **Policy Loss**: Perda do ator (incentiva ações vantajosas)
- **Value Loss**: Perda do crítico (erro na predição V(s))
- **Entropy Loss**: Medida de exploração da política
- **Advantage**: Indicativo de qualidade das ações
- **Explained Variance**: Qualidade das predições do crítico
- **Recompensa por Episódio**: Performance do agente
- **Duração do Episódio**: Número de passos até terminar

## 🔧 Configurações Experimentais

### Experimentos DQN:
1. **Baseline**: Configuração padrão (`dqn_config.yaml`)
2. **Learning Rate Baixo**: LR = 0.0001 (`dqn_config_lr_low.yaml`)
3. **Learning Rate Alto**: LR = 0.01 (`dqn_config_lr_high.yaml`)
4. **Gamma Baixo**: γ = 0.95 (`dqn_config_gamma_low.yaml`)
5. **Rede Simples**: [64] neurônios (`dqn_config_simple_net.yaml`)

### Experimentos A3C:
1. **Baseline**: Configuração padrão (`a3c_config.yaml`)
2. **Learning Rate Baixo**: LR = 0.00001 (`a3c_config_lr_low.yaml`)
3. **Entropy Zero**: entropy_coef = 0.0 (`a3c_config_entropy_zero.yaml`)

## 🚀 Como Executar

### Treinamento Individual

```bash
# DQN com configuração padrão
cd scripts
python train.py --agent dqn

# A3C com configuração padrão  
python train.py --agent a3c
```

### Executar Todos os Experimentos

```bash
cd scripts
python run_experiments.py
```

### Visualizar Resultados no TensorBoard

```bash
# Visualizar todos os experimentos
tensorboard --logdir results/logs

# Visualizar experimento específico
tensorboard --logdir results/logs/DQN_lr_0.01
```

### Análise Automatizada

```bash
cd scripts
python analyze_results.py
```

## 📈 Interpretação das Métricas

### Loss (DQN)
- **Diminuindo**: ✓ Rede está convergindo
- **Oscilando**: ⚠️ Possível instabilidade na taxa de aprendizado
- **Aumentando**: ✗ Problema na configuração

### Policy Loss (A3C)
- **Negativa**: Normal (maximização da vantagem)
- **Diminuindo em módulo**: ✓ Política melhorando
- **Muito instável**: ⚠️ LR muito alto

### Value Loss (A3C)
- **Diminuindo**: ✓ Crítico aprendendo a predizer retornos
- **Estagnada**: ⚠️ Crítico não está aprendendo

### Entropy Loss (A3C)
- **Negativa**: Normal (entropia é positiva, loss é negativa)
- **Próxima de zero**: ⚠️ Baixa exploração
- **Muito negativa**: ✓ Alta exploração

### Explained Variance (A3C)
- **Próxima de 1**: ✓ Crítico prediz bem os retornos
- **Próxima de 0**: ⚠️ Crítico não está aprendendo
- **Negativa**: ✗ Crítico pior que baseline

## 🔍 Análise Comparativa

### TensorBoard - Comparação de Experimentos:
1. Abra o TensorBoard: `tensorboard --logdir results/logs`
2. Use as abas para comparar métricas
3. Selecione múltiplos experimentos para sobreposição
4. Analise correlações entre métricas

### Perguntas-Chave para Análise:

1. **A loss está convergindo?** Observe se diminui consistentemente
2. **Existe correlação entre loss e recompensa?** Loss estável → recompensa estável?
3. **O epsilon está decaindo adequadamente?** (DQN)
4. **A entropia está adequada para exploração?** (A3C)
5. **O crítico está aprendendo?** Explained variance melhorando? (A3C)

## 📊 Estrutura dos Resultados

```
results/
├── logs/                     # Logs do TensorBoard
│   ├── DQN/                 # Baseline DQN
│   ├── DQN_lr_0.01/         # DQN LR alto
│   ├── DQN_lr_0.0001/       # DQN LR baixo
│   ├── A3C/                 # Baseline A3C
│   └── ...
├── models/                   # Modelos salvos
└── analysis/                 # Análises geradas
    ├── dqn_comparison.png
    ├── a3c_comparison.png
    └── experiment_report.md
```

## 🎯 Objetivos dos Experimentos

1. **Learning Rate**: Encontrar o equilíbrio entre velocidade e estabilidade
2. **Gamma**: Avaliar o impacto da "visão de futuro"
3. **Arquitetura**: Determinar complexidade adequada da rede
4. **Target Update Frequency**: Estabilidade do Q-learning (DQN)
5. **Entropy Coefficient**: Balanceamento exploração vs. exploitation (A3C)

## 🔬 Hipóteses Testadas

- **LR Alto**: ⚠️ Pode causar instabilidade e oscilações
- **LR Baixo**: ⚠️ Convergência muito lenta
- **Gamma Baixo**: 🎯 Foco em recompensas imediatas
- **Rede Simples**: ⚡ Pode ser suficiente para problemas simples
- **Entropy Zero**: 🚫 Reduz exploração drasticamente

## 📝 Relatório de Análise

Após executar os experimentos, um relatório detalhado é gerado automaticamente em:
`results/analysis/experiment_report.md`

Este relatório inclui:
- Comparação de performance entre experimentos
- Análise das métricas implementadas
- Conclusões sobre os hiperparâmetros testados
- Recomendações para futuros experimentos

## 🛠️ Dependências

```
gymnasium[classic_control]
torch
tensorboard
numpy
matplotlib
pandas
```

## 🎓 Próximos Passos

1. Implementar Prioritized Experience Replay (DQN)
2. Testar diferentes arquiteturas (LSTM, Attention)
3. Experimentos com diferentes ambientes
4. Implementar algoritmos mais avançados (PPO, SAC)
5. Hyperparameter optimization automatizado