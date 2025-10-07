#!/bin/bash

# Script para monitorar o progresso dos experimentos

echo "=== MONITORAMENTO DE EXPERIMENTOS ==="
echo "Arquivo de log: experiment_output.log"
echo "Para sair, pressione Ctrl+C"
echo "======================================="
echo

# Verificar se o processo está rodando
if pgrep -f "run_experiments.py" > /dev/null; then
    echo "✓ Experimentos rodando em background"
else
    echo "✗ Nenhum experimento rodando"
fi

echo
echo "=== ÚLTIMAS LINHAS DO LOG ==="
tail -20 experiment_output.log

echo
echo "=== MONITORAMENTO EM TEMPO REAL ==="
echo "Pressione Ctrl+C para sair..."
tail -f experiment_output.log