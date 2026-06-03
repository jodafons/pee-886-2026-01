import os
import re
import glob
import numpy as np
import scipy.stats as stats
from matplotlib import rc_file
from matplotlib import pyplot as plt

# Mantendo sua configuração de estilo
rc_file('~/guiaraujo_medium.mplstyle') 
# (Descomente a linha acima caso o arquivo de estilo esteja acessível no ambiente em que for rodar)

def get_acc_from_file_path(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    matches_accuracy = re.findall(r"'accuracy':\s*'([^']+)'", content)
    return [float(valor) for valor in matches_accuracy]

def get_info_from_file_path(path):
    # Get file name from path de forma agnóstica ao sistema operacional
    file_name = os.path.basename(path)

    # Example file name 1: mnist_4cli_2.log (run 2)
    # Default num of local epochs is 2 if not specified in the file name
    file_name_parts = file_name.split('_')
    if len(file_name_parts) == 3:        
        num_clients = int(file_name_parts[1].replace('cli', ''))
        local_epochs = 2
        return num_clients, local_epochs
    
    # Example file name: mnist_4cli_4epoch_2.log (run 2)
    else:
        num_clients = int(file_name_parts[1].replace('cli', ''))
        local_epochs = int(file_name_parts[2].replace('epoch', ''))
        return num_clients, local_epochs

def plot_federated_learning_results(directory_path='.'):
    # Busca apenas arquivos que iniciam com "mnist" e terminam com ".log"
    search_pattern = os.path.join(directory_path, 'mnist*.log')
    log_files = glob.glob(search_pattern)
    
    # Dicionário para agrupar os resultados por cenário
    # Chave: (num_clients, local_epochs) -> Valor: lista de listas de acurácia
    scenarios = {}
    
    for path in log_files:
        # Pula diretórios caso existam com esse padrão
        if not os.path.isfile(path):
            continue
            
        num_clients, local_epochs = get_info_from_file_path(path)
        accuracies = get_acc_from_file_path(path)
        
        scenario_key = (num_clients, local_epochs)
        
        if scenario_key not in scenarios:
            scenarios[scenario_key] = []
        scenarios[scenario_key].append(accuracies)
        
    plt.figure(figsize=(8, 6))
    
    # Processa e plota cada cenário
    for (clientes, epocas), lista_acuracias in scenarios.items():
        # Converte para array numpy. 
        # Garantindo que todas as execuções tenham o mesmo tamanho (truncando pela menor, caso haja diferença nas rodadas)
        min_rounds = min(len(acc) for acc in lista_acuracias)
        acc_matrix = np.array([acc[:min_rounds] for acc in lista_acuracias])
        
        # Eixo x (Rodadas Globais), começando de 1
        x_rounds = np.arange(1, min_rounds + 1)
        
        # Média e desvio padrão amostral (ddof=1) no eixo 0 (ao longo das 5 execuções)
        mean_acc = np.mean(acc_matrix, axis=0)
        std_acc = np.std(acc_matrix, axis=0, ddof=1)
        
        n_execucoes = acc_matrix.shape[0]
        
        # Cálculo do Intervalo de Confiança de 50% usando t-Student
        confidence_level = 0.95
        # Graus de liberdade
        df = n_execucoes - 1 
        # Valor crítico t
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df) 
        # Margem de erro
        margin_of_error = t_critical * (std_acc / np.sqrt(n_execucoes))
        
        label_name = f"{clientes} Clientes, {epocas} Épocas Locais"
        
        # Plot do gráfico de linhas com as barras de erro
        plt.errorbar(x_rounds, mean_acc, yerr=margin_of_error, label=label_name, 
                     marker='o')

    # Configurações do gráfico
    plt.xlabel('Rodadas Globais')
    plt.ylabel('Acurácia Média')
    plt.legend(loc='lower right')
    #plt.grid(True, linestyle='--', alpha=0.7)
    
    # Sem título conforme solicitado
    plt.tight_layout()
    plt.savefig('mnist.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # Substitua '.' pelo caminho do seu diretório caso os logs não estejam na mesma pasta do script
    diretorio_dos_logs = '.' 
    plot_federated_learning_results(diretorio_dos_logs)