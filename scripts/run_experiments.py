"""
Script para executar experimentos automaticamente com diferentes configurações
"""
import subprocess
import os
import time
import sys

# Forçar flush do stdout/stderr para logs em tempo real
sys.stdout.flush()
sys.stderr.flush()

def run_experiment(agent_type, config_file, experiment_name):
    """Executa um experimento com a configuração especificada"""
    print(f"\n{'='*60}")
    print(f"Iniciando experimento: {experiment_name}")
    print(f"Agente: {agent_type}")
    print(f"Configuração: {config_file}")
    print(f"{'='*60}")
    sys.stdout.flush()  # Forçar flush
    
    # Comando para executar o treinamento
    cmd = [
        "python", "-u", "scripts/train.py",  # -u para unbuffered output
        "--agent", agent_type,
        "--config", config_file
    ]
    
    try:
        # Executar treinamento diretamente com o arquivo de configuração específico
        start_time = time.time()
        print(f"Executando comando: {' '.join(cmd)}")
        sys.stdout.flush()
        
        # Usar subprocess.Popen para output em tempo real
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                   universal_newlines=True, bufsize=1)
        
        # Mostrar output em tempo real
        for line in process.stdout:
            print(line.rstrip())
            sys.stdout.flush()
        
        process.wait()
        end_time = time.time()
        
        print(f"Experimento concluído em {end_time - start_time:.2f} segundos")
        sys.stdout.flush()
        
        if process.returncode == 0:
            print("✓ Experimento executado com sucesso!")
        else:
            print(f"✗ Erro durante a execução (código: {process.returncode})")
            
        sys.stdout.flush()
            
    except Exception as e:
        print(f"✗ Erro ao executar comando: {e}")
        sys.stdout.flush()

def main():
    """Executa todos os experimentos"""
    experiments = [
        ("dqn", "configs/dqn_config_lr_low.yaml", "DQN - Learning Rate Baixo (0.0001)"),
        ("dqn", "configs/dqn_config_lr_high.yaml", "DQN - Learning Rate Alto (0.01)"),
        ("dqn", "configs/dqn_config_gamma_low.yaml", "DQN - Gamma Baixo (0.95)"),
        ("dqn", "configs/dqn_config_simple_net.yaml", "DQN - Rede Simples"),
        ("a3c", "configs/a3c_config_lr_low.yaml", "A3C - Learning Rate Baixo (0.00001)"),
        ("a3c", "configs/a3c_config_entropy_zero.yaml", "A3C - Entropy Coef Zero"),
    ]
    
    print("Iniciando bateria de experimentos...")
    print(f"Total de experimentos: {len(experiments)}")
    sys.stdout.flush()
    
    for i, (agent, config, name) in enumerate(experiments, 1):
        print(f"\nExperimento {i}/{len(experiments)}")
        sys.stdout.flush()
        run_experiment(agent, config, name)
        
        # Pausa entre experimentos
        if i < len(experiments):
            print(f"\nAguardando 30 segundos antes do próximo experimento...")
            sys.stdout.flush()
            time.sleep(30)
    
    print("\n" + "="*60)
    print("TODOS OS EXPERIMENTOS CONCLUÍDOS!")
    print("="*60)
    print("\nPara visualizar os resultados, execute:")
    print("tensorboard --logdir results/logs")
    sys.stdout.flush()

if __name__ == "__main__":
    # Mudar para o diretório raiz do projeto
    os.chdir("/app")
    main()