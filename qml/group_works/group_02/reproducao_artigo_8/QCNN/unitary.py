# This module contains the set of unitary ansatze that will be used to benchmark the performances of Quantum Convolutional Neural Network (QCNN) in QCNN.ipynb module
import pennylane as qml
from typing import List, Union
import numpy as np

# Unitary Ansatze for Convolutional Layer

def U_TTN(params: Union[List[float], np.ndarray], wires: List[int]) -> None:
    """
    Aplica o ansatz unitário inspirado em Tensor Tree Networks (TTN).
    
    Consiste em rotações RY independentes seguidas por uma porta CNOT para emaranhamento.

    Args:
        params (list ou ndarray): Lista contendo 2 parâmetros angulares.
        wires (list de int): Lista contendo os índices de 2 qubits alvo.

    Returns:
        None: As operações são aplicadas in-place no circuito.
    """
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def U_5(params: Union[List[float], np.ndarray], wires: List[int]) -> None:
    """
    Aplica o ansatz unitário de circuito parametrizado número 5 (U_5).
    
    Utiliza rotações RX e RZ intercaladas com portas CRZ cruzadas para gerar emaranhamento.

    Args:
        params (list ou ndarray): Lista contendo 10 parâmetros angulares.
        wires (list de int): Lista contendo os índices de 2 qubits alvo.

    Returns:
        None
    """
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.CRZ(params[4], wires=[wires[1], wires[0]])
    qml.CRZ(params[5], wires=[wires[0], wires[1]])
    qml.RX(params[6], wires=wires[0])
    qml.RX(params[7], wires=wires[1])
    qml.RZ(params[8], wires=wires[0])
    qml.RZ(params[9], wires=wires[1])


def U_6(params: Union[List[float], np.ndarray], wires: List[int]) -> None:
    """
    Aplica o ansatz unitário de circuito parametrizado número 6 (U_6).
    
    Semelhante ao U_5, mas substitui o emaranhamento de fase (CRZ) por rotações controladas em X (CRX).

    Args:
        params (list ou ndarray): Lista contendo 10 parâmetros angulares.
        wires (list de int): Lista contendo os índices de 2 qubits alvo.

    Returns:
        None
    """
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.CRX(params[4], wires=[wires[1], wires[0]])
    qml.CRX(params[5], wires=[wires[0], wires[1]])
    qml.RX(params[6], wires=wires[0])
    qml.RX(params[7], wires=wires[1])
    qml.RZ(params[8], wires=wires[0])
    qml.RZ(params[9], wires=wires[1])


def U_9(params: Union[List[float], np.ndarray], wires: List[int]) -> None:
    """
    Aplica o ansatz unitário de circuito parametrizado número 9 (U_9).
    
    Cria uma superposição inicial com portas Hadamard, emaranha com CZ, e aplica 
    rotações parametrizadas RX em cada qubit.

    Args:
        params (list ou ndarray): Lista contendo 2 parâmetros angulares.
        wires (list de int): Lista contendo os índices de 2 qubits alvo.

    Returns:
        None
    """
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])
    qml.CZ(wires=[wires[0], wires[1]])
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])


def U_13(params: Union[List[float], np.ndarray], wires: List[int]) -> None:
    """
    Aplica o ansatz unitário de circuito parametrizado número 13 (U_13).

    Args:
        params (list ou ndarray): Lista contendo 6 parâmetros angulares.
        wires (list de int): Lista contendo os índices de 2 qubits alvo.

    Returns:
        None
    """
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CRZ(params[2], wires=[wires[1], wires[0]])
    qml.RY(params[3], wires=wires[0])
    qml.RY(params[4], wires=wires[1])
    qml.CRZ(params[5], wires=[wires[0], wires[1]])


def U_14(params: Union[List[float], np.ndarray], wires: List[int]) -> None:
    """
    Aplica o ansatz unitário de circuito parametrizado número 14 (U_14).

    Args:
        params (list ou ndarray): Lista contendo 6 parâmetros angulares.
        wires (list de int): Lista contendo os índices de 2 qubits alvo.

    Returns:
        None
    """
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CRX(params[2], wires=[wires[1], wires[0]])
    qml.RY(params[3], wires=wires[0])
    qml.RY(params[4], wires=wires[1])
    qml.CRX(params[5], wires=[wires[0], wires[1]])


def U_15(params: Union[List[float], np.ndarray], wires: List[int]) -> None:
    """
    Aplica o ansatz unitário de circuito parametrizado número 15 (U_15).

    Args:
        params (list ou ndarray): Lista contendo 4 parâmetros angulares.
        wires (list de int): Lista contendo os índices de 2 qubits alvo.

    Returns:
        None
    """
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def U_SO4(params: Union[List[float], np.ndarray], wires: List[int]) -> None:
    """
    Aplica uma parametrização que mapeia para o grupo especial ortogonal SO(4).
    
    Capaz de expressar qualquer operação ortogonal de 2 qubits preservando a paridade real.

    Args:
        params (list ou ndarray): Lista contendo 6 parâmetros angulares.
        wires (list de int): Lista contendo os índices de 2 qubits alvo.

    Returns:
        None
    """
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[4], wires=wires[0])
    qml.RY(params[5], wires=wires[1])


def U_SU4(params: Union[List[float], np.ndarray], wires: List[int]) -> None:
    """
    Aplica uma parametrização universal de 2 qubits capaz de expressar 
    qualquer operação unitária no grupo SU(4).
    
    Usa portas U3 arbitrárias e 3 CNOTs (o mínimo necessário para universalidade de 2 qubits).

    Args:
        params (list ou ndarray): Lista contendo 15 parâmetros angulares complexos/reais.
        wires (list de int): Lista contendo os índices de 2 qubits alvo.

    Returns:
        None
    """
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])

# Pooling Layer

def Pooling_ansatz1(params: Union[List[float], np.ndarray], wires: List[int]) -> None:
    """
    Aplica o primeiro ansatz de camada de agrupamento (Pooling) para redução dimensional.
    
    Atua controlando a fase de um qubit com base no outro, inverte o qubit de controle,
    e aplica uma rotação condicional em X. Geralmente, um dos qubits é medido/descartado após isso.

    Args:
        params (list ou ndarray): Lista contendo 2 parâmetros angulares.
        wires (list de int): Lista contendo os índices de 2 qubits alvo.

    Returns:
        None
    """
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])


def Pooling_ansatz2(wires: List[int]) -> None:
    """
    Aplica o segundo ansatz de camada de agrupamento (Pooling).
    
    Nota de Depuração: No código original submetido, `qml.CRZ` foi chamado sem 
    parâmetros angulares. Nas versões estáveis do PennyLane, a porta CRZ requer 
    um ângulo (`phi`). Se o objetivo for apenas uma porta controlada não parametrizada, 
    provavelmente a intenção original era `qml.CZ(wires=[wires[0], wires[1]])`.

    Args:
        wires (list de int): Lista contendo os índices de 2 qubits alvo.

    Returns:
        None
    """
    # Mantido como no original, mas fica o alerta sobre possível erro do PennyLane.
    #qml.CRZ(wires=[wires[0], wires[1]])
    qml.CZ(wires=[wires[0], wires[1]])


def Pooling_ansatz3(*params: float, wires: List[int]) -> None:
    """
    Aplica o terceiro ansatz de camada de agrupamento (Pooling).
    
    Utiliza uma rotação controlada genérica (CRot) que recebe 3 ângulos.
    O desempacotamento de parâmetros `*params` exige que os ângulos sejam passados 
    diretamente como argumentos, não como uma lista única.

    Args:
        *params (float): Três parâmetros angulares sequenciais (phi, theta, omega).
        wires (list de int): Lista contendo os índices de 2 qubits alvo passados como keyword argument.

    Returns:
        None
    """
    qml.CRot(*params, wires=[wires[0], wires[1]])