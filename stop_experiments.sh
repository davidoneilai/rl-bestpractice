#!/bin/bash

# Script para parar os experimentos em execução

echo "=== PARANDO EXPERIMENTOS ==="

# Encontrar e matar processos relacionados aos experimentos
PIDS=$(pgrep -f "run_experiments.py")

if [ -z "$PIDS" ]; then
    echo "Nenhum experimento rodando."
else
    echo "Parando processos: $PIDS"
    kill $PIDS
    sleep 2
    
    # Verificar se ainda estão rodando
    REMAINING=$(pgrep -f "run_experiments.py")
    if [ -z "$REMAINING" ]; then
        echo "✓ Experimentos parados com sucesso!"
    else
        echo "Forçando parada dos processos restantes..."
        kill -9 $REMAINING
        echo "✓ Experimentos forçadamente parados!"
    fi
fi

echo "=== ÚLTIMA SAÍDA DO LOG ==="
tail -10 experiment_output.log 2>/dev/null || echo "Arquivo de log não encontrado."