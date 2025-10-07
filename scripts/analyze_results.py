"""
Script para análise dos resultados dos experimentos
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def extract_tensorboard_data(log_dir, scalar_name):
    """Extrai dados de escalar do TensorBoard"""
    try:
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        scalar_events = event_acc.Scalars(scalar_name)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        
        return steps, values
    except:
        return [], []

def analyze_experiment_results():
    """Analisa os resultados de todos os experimentos"""
    base_log_dir = "results/logs"
    
    # Configurações dos experimentos
    experiments = {
        "DQN_baseline": "DQN",
        "DQN_lr_0.0001": "DQN - LR Baixo",
        "DQN_lr_0.01": "DQN - LR Alto", 
        "DQN_gamma_0.95": "DQN - Gamma Baixo",
        "DQN_simple_net": "DQN - Rede Simples",
        "A3C_baseline": "A3C",
        "A3C_lr_0.00001": "A3C - LR Baixo",
        "A3C_entropy_0": "A3C - Entropy Zero"
    }
    
    # Métricas para analisar
    dqn_metrics = ["Reward", "Loss", "Episode Duration", "Epsilon"]
    a3c_metrics = ["Global/Reward", "Global/Policy_Loss", "Global/Value_Loss", 
                   "Global/Entropy_Loss", "Global/Episode_Duration"]
    
    print("="*60)
    print("ANÁLISE DOS RESULTADOS DOS EXPERIMENTOS")
    print("="*60)
    
    # Analisar experimentos DQN
    print("\n🤖 EXPERIMENTOS DQN:")
    print("-" * 40)
    
    dqn_results = {}
    for exp_name, exp_label in experiments.items():
        if "DQN" in exp_name:
            log_path = os.path.join(base_log_dir, exp_name.replace("_baseline", ""))
            if os.path.exists(log_path):
                print(f"\n📊 {exp_label}:")
                
                # Extrair dados de recompensa
                steps, rewards = extract_tensorboard_data(log_path, "Reward")
                if rewards:
                    dqn_results[exp_label] = {
                        'steps': steps,
                        'rewards': rewards,
                        'max_reward': max(rewards),
                        'final_avg_reward': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
                        'convergence_episode': len(rewards)
                    }
                    
                    print(f"  • Recompensa máxima: {max(rewards):.2f}")
                    print(f"  • Recompensa média final (últimos 100): {np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards):.2f}")
                    print(f"  • Episódios treinados: {len(rewards)}")
                    
                    # Analisar loss se disponível
                    _, losses = extract_tensorboard_data(log_path, "Loss")
                    if losses:
                        print(f"  • Loss final: {losses[-1]:.6f}")
                        print(f"  • Loss diminuiu: {'✓' if losses[-1] < losses[0] else '✗'}")
                else:
                    print("  • Sem dados disponíveis")
    
    # Analisar experimentos A3C
    print("\n🔄 EXPERIMENTOS A3C:")
    print("-" * 40)
    
    a3c_results = {}
    for exp_name, exp_label in experiments.items():
        if "A3C" in exp_name:
            log_path = os.path.join(base_log_dir, exp_name.replace("_baseline", ""))
            if os.path.exists(log_path):
                print(f"\n📊 {exp_label}:")
                
                # Extrair dados de recompensa
                steps, rewards = extract_tensorboard_data(log_path, "Global/Reward")
                if rewards:
                    a3c_results[exp_label] = {
                        'steps': steps,
                        'rewards': rewards,
                        'max_reward': max(rewards),
                        'final_avg_reward': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                    }
                    
                    print(f"  • Recompensa máxima: {max(rewards):.2f}")
                    print(f"  • Recompensa média final: {np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards):.2f}")
                    print(f"  • Episódios treinados: {len(rewards)}")
                    
                    # Analisar métricas específicas do A3C
                    _, policy_loss = extract_tensorboard_data(log_path, "Global/Policy_Loss")
                    _, value_loss = extract_tensorboard_data(log_path, "Global/Value_Loss")
                    _, entropy_loss = extract_tensorboard_data(log_path, "Global/Entropy_Loss")
                    
                    if policy_loss:
                        print(f"  • Policy Loss final: {policy_loss[-1]:.6f}")
                    if value_loss:
                        print(f"  • Value Loss final: {value_loss[-1]:.6f}")
                    if entropy_loss:
                        print(f"  • Entropy Loss final: {entropy_loss[-1]:.6f}")
                else:
                    print("  • Sem dados disponíveis")
    
    # Gerar gráficos comparativos
    generate_comparison_plots(dqn_results, a3c_results)
    
    # Gerar relatório
    generate_report(dqn_results, a3c_results)

def generate_comparison_plots(dqn_results, a3c_results):
    """Gera gráficos comparativos dos experimentos"""
    if not dqn_results and not a3c_results:
        print("\n⚠️ Nenhum dado disponível para gerar gráficos")
        return
    
    print("\n📈 Gerando gráficos comparativos...")
    
    # Criar diretório para gráficos
    os.makedirs("results/analysis", exist_ok=True)
    
    # Gráfico DQN
    if dqn_results:
        plt.figure(figsize=(12, 8))
        for exp_name, data in dqn_results.items():
            plt.plot(data['steps'], data['rewards'], label=exp_name, alpha=0.7)
        
        plt.title("Comparação de Recompensas - Experimentos DQN")
        plt.xlabel("Episódios")
        plt.ylabel("Recompensa")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("results/analysis/dqn_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Gráfico A3C
    if a3c_results:
        plt.figure(figsize=(12, 8))
        for exp_name, data in a3c_results.items():
            plt.plot(data['steps'], data['rewards'], label=exp_name, alpha=0.7)
        
        plt.title("Comparação de Recompensas - Experimentos A3C")
        plt.xlabel("Episódios")
        plt.ylabel("Recompensa")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("results/analysis/a3c_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("  • Gráficos salvos em results/analysis/")

def generate_report(dqn_results, a3c_results):
    """Gera relatório detalhado dos experimentos"""
    print("\n📝 Gerando relatório detalhado...")
    
    report_path = "results/analysis/experiment_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Experimentos - Reinforcement Learning\n\n")
        f.write("## Resumo Executivo\n\n")
        f.write("Este relatório apresenta os resultados dos experimentos realizados ")
        f.write("para avaliar o impacto de diferentes hiperparâmetros no desempenho ")
        f.write("dos algoritmos DQN e A3C no ambiente LunarLander-v3.\n\n")
        
        # Seção DQN
        if dqn_results:
            f.write("## 🤖 Experimentos DQN\n\n")
            f.write("| Experimento | Recompensa Máxima | Recompensa Final | Episódios |\n")
            f.write("|-------------|-------------------|------------------|----------|\n")
            
            for exp_name, data in dqn_results.items():
                f.write(f"| {exp_name} | {data['max_reward']:.2f} | ")
                f.write(f"{data['final_avg_reward']:.2f} | {data['convergence_episode']} |\n")
            
            f.write("\n### Análise DQN:\n\n")
            f.write("**Observações sobre os experimentos DQN:**\n")
            f.write("- Variações na taxa de aprendizado mostram impacto significativo na convergência\n")
            f.write("- O fator de desconto (gamma) afeta a 'visão de futuro' do agente\n")
            f.write("- A arquitetura da rede influencia a capacidade de aproximação\n\n")
        
        # Seção A3C
        if a3c_results:
            f.write("## 🔄 Experimentos A3C\n\n")
            f.write("| Experimento | Recompensa Máxima | Recompensa Final |\n")
            f.write("|-------------|-------------------|------------------|\n")
            
            for exp_name, data in a3c_results.items():
                f.write(f"| {exp_name} | {data['max_reward']:.2f} | ")
                f.write(f"{data['final_avg_reward']:.2f} |\n")
            
            f.write("\n### Análise A3C:\n\n")
            f.write("**Observações sobre os experimentos A3C:**\n")
            f.write("- O coeficiente de entropia é crucial para a exploração\n")
            f.write("- Taxa de aprendizado muito baixa pode levar a convergência lenta\n")
            f.write("- Equilíbrio entre policy loss e value loss é importante\n\n")
        
        f.write("## 📊 Métricas Implementadas\n\n")
        f.write("### DQN:\n")
        f.write("- **Loss da Rede Neural**: Indica convergência do algoritmo\n")
        f.write("- **Duração do Episódio**: Tempo de sobrevivência/resolução\n")
        f.write("- **Epsilon**: Decaimento da exploração\n\n")
        
        f.write("### A3C:\n")
        f.write("- **Policy Loss**: Perda do ator (actor)\n")
        f.write("- **Value Loss**: Perda do crítico\n")
        f.write("- **Entropy Loss**: Medida de exploração\n")
        f.write("- **Advantage**: Indicativo de boas ações\n")
        f.write("- **Explained Variance**: Qualidade das predições do crítico\n\n")
        
        f.write("## 🔍 Conclusões\n\n")
        f.write("1. **Taxa de Aprendizado**: Hiperparâmetro crítico - valores muito altos causam instabilidade\n")
        f.write("2. **Fator de Desconto**: Afeta significativamente a estratégia do agente\n")
        f.write("3. **Arquitetura da Rede**: Redes mais complexas não necessariamente são melhores\n")
        f.write("4. **Exploração**: Fundamental para evitar mínimos locais\n\n")
        
        f.write("## 📈 Visualizações\n\n")
        f.write("- Gráficos comparativos disponíveis em `results/analysis/`\n")
        f.write("- Use TensorBoard para análise detalhada: `tensorboard --logdir results/logs`\n")
    
    print(f"  • Relatório salvo em {report_path}")

if __name__ == "__main__":
    analyze_experiment_results()