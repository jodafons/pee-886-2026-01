import pennylane as qml
from typing import List, Union
import numpy as np

def Angular_Hybrid_2(X: Union[List[float], np.ndarray], wires: List[int]) -> None:
    """
    Prepara um estado quântico de 2 qubits codificando 3 características features 
    diretamente nos ângulos de uma estrutura do tipo Möttönen.

    Args:
        X (list ou ndarray): Um vetor contendo exatamente 3 valores angulares (features).
        wires (list de int): Uma lista com 2 índices de qubits (ex: [0, 1]).

    Returns:
        None: As operações são aplicadas in-place no circuito PennyLane.
    """
    qml.RY(X[0], wires=wires[0])

    qml.PauliX(wires=wires[0])
    qml.CRY(X[1], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRY(X[2], wires=[wires[0], wires[1]])


def Angular_Hybrid_4(X: Union[List[float], np.ndarray], wires: List[int]) -> None:
    """
    Prepara um estado quântico de 4 qubits codificando 15 características clássicas 
    através de uma cascata de rotações uniformemente controladas (multiplexadores).

    Essa função implementa a decomposição de hardware exata de uma Preparação de Estado 
    de Möttönen, mas alimentando os dados clássicos diretamente como os ângulos das 
    portas RY, em vez de calcular os ângulos a partir de amplitudes normalizadas.

    Args:
        X (list ou ndarray): Um vetor contendo exatamente 15 valores angulares (features).
        wires (list de int): Uma lista com 4 índices de qubits (ex: [0, 1, 2, 3]).

    Returns:
        None: As operações são aplicadas in-place no circuito PennyLane.
    """
    # 1º Bloco: Análogo ao Angular_Hybrid_2 (3 parâmetros, controla os fios 0 e 1)
    qml.RY(X[0], wires=wires[0])

    qml.PauliX(wires=wires[0])
    qml.CRY(X[1], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRY(X[2], wires=[wires[0], wires[1]])

    # 2º Bloco: Rotações no fio 2, controladas pelos fios 0 e 1 (4 parâmetros)
    qml.RY(X[3], wires=wires[2])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.RY(X[4], wires=wires[2])
    qml.CNOT(wires=[wires[0], wires[2]])
    qml.RY(X[5], wires=wires[2])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.RY(X[6], wires=wires[2])
    qml.CNOT(wires=[wires[0], wires[2]])

    # 3º Bloco: Rotações no fio 3, controladas pelos fios 0, 1 e 2 (8 parâmetros)
    qml.RY(X[7], wires=wires[3])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.RY(X[8], wires=wires[3])
    qml.CNOT(wires=[wires[1], wires[3]])
    qml.RY(X[9], wires=wires[3])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.RY(X[10], wires=wires[3])
    qml.CNOT(wires=[wires[0], wires[3]])
    qml.RY(X[11], wires=wires[3])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.RY(X[12], wires=wires[3])
    qml.CNOT(wires=[wires[1], wires[3]])
    qml.RY(X[13], wires=wires[3])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.RY(X[14], wires=wires[3])
    qml.CNOT(wires=[wires[0], wires[3]])