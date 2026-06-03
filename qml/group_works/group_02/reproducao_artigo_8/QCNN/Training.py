# Implementation of Quantum circuit training procedure
import QCNN_circuit
import Hierarchical_circuit
import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as anp
from typing import List, Tuple, Union, Callable

def square_loss(labels: List[Union[int, float]], predictions: List[float]) -> float:
    """
    Calcula o Erro Quadrático Médio (MSE - Mean Squared Error) entre os rótulos 
    verdadeiros e as predições do circuito quântico.

    Args:
        labels (list): Lista de rótulos verdadeiros.
        predictions (list): Lista de valores contínuos preditos pelo circuito 
                            (geralmente o valor esperado de uma medição Pauli-Z).

    Returns:
        float: O valor do erro quadrático médio.
    """
    loss = 0.0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

def cross_entropy(labels: List[int], predictions: List[Union[List[float], np.ndarray]]) -> float:
    """
    Calcula a Perda de Entropia Cruzada Binária (BCE - Binary Cross-Entropy).

    Args:
        labels (list de int): Lista de rótulos verdadeiros (0 ou 1).
        predictions (list de arrays): Lista contendo as distribuições de probabilidade 
            retornadas pelo circuito. Para um problema binário, espera-se que cada 
            predição seja um array com [Probabilidade(|0>), Probabilidade(|1>)].

    Returns:
        float: O valor da entropia cruzada calculado utilizando autograd.numpy (anp).
    """
    loss = 0.0
    for l, p in zip(labels, predictions):
        # p[l] pega a probabilidade associada à classe correta (l).
        # p[1 - l] pega a probabilidade da classe incorreta.
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1.0 * loss

def cost(params: np.ndarray, X: List[np.ndarray], Y: List[int], U: str, U_params: int, 
         embedding_type: str, circuit: str, cost_fn: str) -> float:
    """
    Avalia um lote (batch) de dados no circuito quântico e retorna o erro total.
    
    Esta função atua como um 'wrapper' (empacotador) que permite ao otimizador 
    calcular o gradiente em relação ao argumento `params`.

    Args:
        params (ndarray): Vetor de parâmetros treináveis atuais.
        X (list de ndarray): Lote de dados de entrada.
        Y (list de int): Lote de rótulos verdadeiros correspondentes.
        U (str): Nome do ansatz unitário utilizado.
        U_params (int): Número de parâmetros por bloco do ansatz.
        embedding_type (str): Estratégia de codificação de dados.
        circuit (str): Tipo da rede ('QCNN' ou 'Hierarchical').
        cost_fn (str): Função de custo a ser utilizada ('mse' ou 'cross_entropy').

    Returns:
        float: O valor numérico da perda para o lote atual.
    """
    if circuit == 'QCNN':
        predictions = [QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]
    elif circuit == 'Hierarchical':
        predictions = [Hierarchical_circuit.Hierarchical_classifier(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]

    if cost_fn == 'mse':
        loss = square_loss(Y, predictions)
    elif cost_fn == 'cross_entropy':
        loss = cross_entropy(Y, predictions)
    return loss

# Circuit training parameters
steps = 200
learning_rate = 0.01
batch_size = 25

def circuit_training(X_train: List[np.ndarray], Y_train: List[int], U: str, U_params: int, 
                     embedding_type: str, circuit: str, cost_fn: str) -> Tuple[List[float], np.ndarray]:
    """
    Executa o laço principal de treinamento do classificador quântico.

    Calcula a contagem total de parâmetros com base na arquitetura, inicializa 
    os pesos aleatoriamente, e utiliza o otimizador Nesterov Momentum para 
    atualizar os pesos utilizando minilotes (mini-batches).

    Args:
        X_train (list): Conjunto completo de dados de treinamento.
        Y_train (list): Conjunto completo de rótulos de treinamento.
        U (str): Ansatz unitário.
        U_params (int): Parâmetros do ansatz.
        embedding_type (str): Estratégia de codificação.
        circuit (str): Arquitetura do circuito ('QCNN' ou 'Hierarchical').
        cost_fn (str): Função de custo ('mse' ou 'cross_entropy').

    Returns:
        tuple: (loss_history, params)
            - loss_history (list): Histórico do valor da função de custo a cada passo.
            - params (ndarray): Vetor com os pesos finais otimizados.
    """
    # 1. Definição Dinâmica do Número de Parâmetros
    if circuit == 'QCNN':
        if U == 'U_SU4_no_pooling' or U == 'U_SU4_1D' or U == 'U_9_1D':
            # Apenas 3 camadas convolucionais
            total_params = U_params * 3
        else:
            # 3 camadas convolucionais + 3 camadas de pooling (2 parâmetros cada = 6)
            total_params = U_params * 3 + 2 * 3
    elif circuit == 'Hierarchical':
        total_params = U_params * 7

    # 2. Inicialização dos Pesos (Requer Gradiente ativado para o Autograd)
    params = np.random.randn(total_params, requires_grad=True)
    
    # 3. Otimizador
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    loss_history = []

    # 4. Loop de Treinamento
    for it in range(steps):
        # Amostragem de Minilote
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        
        # Passo de otimização
        params, cost_new = opt.step_and_cost(
            lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn),
            params
        )
        
        loss_history.append(cost_new)
        if it % 10 == 0:
            print("iteration: ", it, " cost: ", cost_new)
            
    return loss_history, params