import pennylane as qml
import unitary
import embedding
from typing import Callable, List, Union
import numpy as np

# Quantum Circuits for Convolutional layers

def conv_layer1(U: Callable, params: Union[List[float], np.ndarray]) -> None:
    """
    Aplica a primeira camada convolucional quântica em um sistema de 8 qubits.
    
    A convolução é translacionalmente invariante, aplicando o mesmo ansatz unitário (U) 
    e os mesmos parâmetros (params) ao longo de pares adjacentes e nas bordas.

    Args:
        U (Callable): Função que define o ansatz unitário (ex: unitary.U_SU4).
        params (list ou ndarray): Parâmetros angulares passados para o ansatz.
    """
    U(params, wires=[0, 7]) # Conexão de contorno (periodic boundary condition)
    for i in range(0, 8, 2):
        U(params, wires=[i, i + 1]) # Pares pares (0-1, 2-3, 4-5, 6-7)
    for i in range(1, 7, 2):
        U(params, wires=[i, i + 1]) # Pares ímpares (1-2, 3-4, 5-6)

def conv_layer2(U: Callable, params: Union[List[float], np.ndarray]) -> None:
    """
    Aplica a segunda camada convolucional quântica nos 4 qubits restantes após o primeiro pooling.
    
    Opera nos qubits de índice par (0, 2, 4, 6), que guardam a informação condensada.

    Args:
        U (Callable): Função ansatz unitário.
        params (list ou ndarray): Parâmetros angulares da camada.
    """
    U(params, wires=[0, 6]) # Conexão de contorno
    U(params, wires=[0, 2])
    U(params, wires=[4, 6])
    U(params, wires=[2, 4])

def conv_layer3(U: Callable, params: Union[List[float], np.ndarray]) -> None:
    """
    Aplica a terceira camada convolucional nos 2 qubits finais (0 e 4) antes do pooling final.

    Args:
        U (Callable): Função ansatz unitário.
        params (list ou ndarray): Parâmetros angulares da camada.
    """
    U(params, wires=[0, 4])


# Quantum Circuits for Pooling layers

def pooling_layer1(V: Callable, params: Union[List[float], np.ndarray]) -> None:
    """
    Aplica a primeira camada de pooling, reduzindo a dimensionalidade do sistema pela metade.
    
    Informações dos qubits ímpares (1, 3, 5, 7) são condensadas nos pares (0, 2, 4, 6). 
    Fisicamente, equivale a emaranhar os pares e implicitamente traçar (descartar) um deles.

    Args:
        V (Callable): Função ansatz de pooling (ex: unitary.Pooling_ansatz1).
        params (list ou ndarray): Parâmetros angulares para o pooling (geralmente 2 params).
    """
    for i in range(0, 8, 2):
        V(params, wires=[i + 1, i]) # Qubit controle (descartado) vs alvo (preservado)

def pooling_layer2(V: Callable, params: Union[List[float], np.ndarray]) -> None:
    """
    Aplica a segunda camada de pooling nos 4 qubits ativos.
    Condensa a informação dos qubits 2 e 6 nos qubits 0 e 4.
    """
    V(params, wires=[2, 0])
    V(params, wires=[6, 4])

def pooling_layer3(V: Callable, params: Union[List[float], np.ndarray]) -> None:
    """
    Aplica a terceira e última camada de pooling.
    Condensa a informação do qubit 0 no qubit 4. O qubit 4 será medido no final.
    """
    V(params, wires=[0, 4])


def QCNN_structure(U: Callable, params: Union[List[float], np.ndarray], U_params: int) -> None:
    """
    Constrói a arquitetura hierárquica completa da QCNN (Convolução -> Pooling iterativos).

    Faz o particionamento do vetor de parâmetros global `params` para alocar os 
    pesos específicos para cada camada de convolução e pooling.

    Args:
        U (Callable): Ansatz unitário utilizado para as convoluções.
        params (ndarray): Vetor 1D com todos os parâmetros treináveis do circuito.
        U_params (int): Número de parâmetros que o ansatz `U` requer por bloco.
    """
    # Fatiamento dos parâmetros para as convoluções (compartilhamento de pesos)
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    
    # Fatiamento dos parâmetros para os poolings (assume-se 2 parâmetros por Pooling_ansatz1)
    param4 = params[3 * U_params: 3 * U_params + 2]
    param5 = params[3 * U_params + 2: 3 * U_params + 4]
    param6 = params[3 * U_params + 4: 3 * U_params + 6]

    # Estrutura em árvore binária invertida
    conv_layer1(U, param1)
    pooling_layer1(unitary.Pooling_ansatz1, param4)
    
    conv_layer2(U, param2)
    pooling_layer2(unitary.Pooling_ansatz1, param5)
    
    conv_layer3(U, param3)
    pooling_layer3(unitary.Pooling_ansatz1, param6)


def QCNN_structure_without_pooling(U: Callable, params: Union[List[float], np.ndarray], U_params: int) -> None:
    """
    Estrutura alternativa de QCNN que aplica apenas as convoluções, sem reduzir 
    o número de qubits do sistema (preserva o espaço de Hilbert de 8 qubits).
    """
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]

    conv_layer1(U, param1)
    conv_layer2(U, param2)
    conv_layer3(U, param3)

def QCNN_1D_circuit(U: Callable, params: Union[List[float], np.ndarray], U_params: int) -> None:
    """
    Aplica uma topologia de convolução restrita a 1 dimensão, sem condições de 
    contorno periódico global e sem hierarquia profunda de pooling.
    """
    param1 = params[0: U_params]
    param2 = params[U_params: 2*U_params]
    param3 = params[2*U_params: 3*U_params]

    for i in range(0, 8, 2):
        U(param1, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(param1, wires=[i, i + 1])

    U(param2, wires=[2,3])
    U(param2, wires=[4,5])
    U(param3, wires=[3,4])

# Inicializa o dispositivo simulador de estado exato para 8 qubits
dev = qml.device('default.qubit', wires = 8)

@qml.qnode(dev)
def QCNN(X: Union[List[float], np.ndarray], params: np.ndarray, U: str, U_params: int, 
         embedding_type: str = 'Amplitude', cost_fn: str = 'cross_entropy'):
    """
    QNode principal que executa o circuito quântico fim-a-fim.

    Integra o embedding de dados clássicos, avalia a topologia solicitada dinamicamente, 
    constrói as camadas parametrizadas e retorna a medida do qubit de saída final.

    Args:
        X (list/ndarray): Dados de entrada a serem codificados.
        params (ndarray): Parâmetros de otimização contínuos (pesos da rede).
        U (str): String com o nome do ansatz selecionado.
        U_params (int): Contagem de parâmetros requerida pelo ansatz.
        embedding_type (str): Tipo de codificação de dados ('Amplitude', 'Angle', etc).
        cost_fn (str): Função de custo alvo ('mse' para valor esperado, 'cross_entropy' para probabilidades).

    Returns:
        float ou np.ndarray: Valor esperado no observável PauliZ (para MSE) ou o 
        vetor de probabilidades do qubit 4 (para Cross-Entropy).
    """

    # Data Embedding
    embedding.data_embedding(X, embedding_type=embedding_type)

    # Quantum Convolutional Neural Network Builder
    if U == 'U_TTN':
        QCNN_structure(unitary.U_TTN, params, U_params)
    elif U == 'U_5':
        QCNN_structure(unitary.U_5, params, U_params)
    elif U == 'U_6':
        QCNN_structure(unitary.U_6, params, U_params)
    elif U == 'U_9':
        QCNN_structure(unitary.U_9, params, U_params)
    elif U == 'U_13':
        QCNN_structure(unitary.U_13, params, U_params)
    elif U == 'U_14':
        QCNN_structure(unitary.U_14, params, U_params)
    elif U == 'U_15':
        QCNN_structure(unitary.U_15, params, U_params)
    elif U == 'U_SO4':
        QCNN_structure(unitary.U_SO4, params, U_params)
    elif U == 'U_SU4':
        QCNN_structure(unitary.U_SU4, params, U_params)
    elif U == 'U_SU4_no_pooling':
        QCNN_structure_without_pooling(unitary.U_SU4, params, U_params)
    elif U == 'U_SU4_1D':
        QCNN_1D_circuit(unitary.U_SU4, params, U_params)
    elif U == 'U_9_1D':
        QCNN_1D_circuit(unitary.U_9, params, U_params)
    else:
        print("Invalid Unitary Ansatze")
        return False

    # Medição final
    if cost_fn == 'mse':
        # Retorna a expectativa entre [-1, 1] no qubit 4
        result = qml.expval(qml.PauliZ(4)) 
    elif cost_fn == 'cross_entropy':
        # Retorna a probabilidade [P(|0>), P(|1>)] no qubit 4
        result = qml.probs(wires=4) 
    
    return result