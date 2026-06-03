import data
import Training
import QCNN_circuit
import Hierarchical_circuit
import numpy as np
import os
from typing import List, Union

def accuracy_test(predictions: List[Union[float, List[float]]], labels: List[int], 
                  cost_fn: str, binary: bool = True) -> float:
    """
    Calcula a precisão (accuracy) das predições do modelo quântico.

    A métrica varia dependendo da função de custo e de como os rótulos 
    foram mapeados no dataset original.

    Args:
        predictions (list): Lista com as saídas preditas pelo modelo.
        labels (list): Lista de rótulos verdadeiros.
        cost_fn (str): A função de custo utilizada ('mse' ou 'cross_entropy').
        binary (bool): Indica o mapeamento das classes. 
            - True: Classes são {-1, 1}
            - False: Classes são {0, 1}

    Returns:
        float: A taxa de acerto (entre 0.0 e 1.0).
    """
    if cost_fn == 'mse':
        if binary == True:
            # Para labels {-1, 1}, se a diferença absoluta for < 1, 
            # significa que o sinal da predição acertou o quadrante correto.
            acc = 0
            for l, p in zip(labels, predictions):
                if np.abs(l - p) < 1:
                    acc += 1
            return acc / len(labels)

        else:
            # Para labels {0, 1}, o limiar de decisão é 0.5.
            acc = 0
            for l, p in zip(labels, predictions):
                if np.abs(l - p) < 0.5:
                    acc += 1
            return acc / len(labels)

    elif cost_fn == 'cross_entropy':
        # Compara as probabilidades inferidas para cada classe.
        acc = 0
        for l, p in zip(labels, predictions):
            if p[0] > p[1]:
                P = 0
            else:
                P = 1
            if P == l:
                acc += 1
        return acc / len(labels)


def Encoding_to_Embedding(Encoding: str) -> str:
    """
    Traduz o nome do método de redução de dimensionalidade clássico (Encoding) 
    para a estratégia de codificação quântica correspondente (Embedding).

    Args:
        Encoding (str): O nome do método de extração de features (ex: 'pca32-1').

    Returns:
        str: O nome do template de embedding esperado pelas funções do PennyLane 
             (ex: 'Amplitude-Hybrid4-1').
    """
    # Amplitude Embedding / Angle Embedding
    if Encoding == 'resize256':
        Embedding = 'Amplitude'
    elif Encoding == 'pca8':
        Embedding = 'Angle'
    elif Encoding == 'autoencoder8':
        Embedding = 'Angle'

    # Amplitude Hybrid Embedding (4 qubit block)
    elif Encoding in ['pca32-1', 'autoencoder32-1']: Embedding = 'Amplitude-Hybrid4-1'
    elif Encoding in ['pca32-2', 'autoencoder32-2']: Embedding = 'Amplitude-Hybrid4-2'
    elif Encoding in ['pca32-3', 'autoencoder32-3']: Embedding = 'Amplitude-Hybrid4-3'
    elif Encoding in ['pca32-4', 'autoencoder32-4']: Embedding = 'Amplitude-Hybrid4-4'

    # Amplitude Hybrid Embedding (2 qubit block)
    elif Encoding in ['pca16-1', 'autoencoder16-1']: Embedding = 'Amplitude-Hybrid2-1'
    elif Encoding in ['pca16-2', 'autoencoder16-2']: Embedding = 'Amplitude-Hybrid2-2'
    elif Encoding in ['pca16-3', 'autoencoder16-3']: Embedding = 'Amplitude-Hybrid2-3'
    elif Encoding in ['pca16-4', 'autoencoder16-4']: Embedding = 'Amplitude-Hybrid2-4'

    # Angular HybridEmbedding (4 qubit block)
    elif Encoding in ['pca30-1', 'autoencoder30-1']: Embedding = 'Angular-Hybrid4-1'
    elif Encoding in ['pca30-2', 'autoencoder30-2']: Embedding = 'Angular-Hybrid4-2'
    elif Encoding in ['pca30-3', 'autoencoder30-3']: Embedding = 'Angular-Hybrid4-3'
    elif Encoding in ['pca30-4', 'autoencoder30-4']: Embedding = 'Angular-Hybrid4-4'

    # Angular HybridEmbedding (2 qubit block)
    elif Encoding in ['pca12-1', 'autoencoder12-1']: Embedding = 'Angular-Hybrid2-1'
    elif Encoding in ['pca12-2', 'autoencoder12-2']: Embedding = 'Angular-Hybrid2-2'
    elif Encoding in ['pca12-3', 'autoencoder12-3']: Embedding = 'Angular-Hybrid2-3'
    elif Encoding in ['pca12-4', 'autoencoder12-4']: Embedding = 'Angular-Hybrid2-4'

    # Two Gates Compact Encoding
    elif Encoding in ['pca16-compact', 'autoencoder16-compact']: 
        Embedding = 'Angle-compact'
        
    return Embedding


def Benchmarking(dataset: str, classes: List[int], Unitaries: List[str], U_num_params: List[int], 
                 Encodings: List[str], circuit: str, cost_fn: str, binary: bool = True) -> None:
    """
    Executa a grade de testes (Grid Search) para avaliar diferentes arquiteturas quânticas.

    Itera sobre combinações de Ansatze unitários e métodos de codificação, treina 
    os modelos, avalia a precisão no conjunto de teste e anexa os resultados em um 
    arquivo de texto.

    Args:
        dataset (str): Nome do dataset ('mnist' ou 'fashion_mnist').
        classes (list): As duas classes a serem filtradas para classificação.
        Unitaries (list de str): Lista com os nomes dos circuitos (ex: ['U_SU4', 'U_9']).
        U_num_params (list de int): Quantidade de parâmetros exigida por cada circuito na lista.
        Encodings (list de str): Lista com os métodos clássicos a testar (ex: ['pca8', 'resize256']).
        circuit (str): Tipo da rede ('QCNN' ou 'Hierarchical').
        cost_fn (str): Função de perda alvo.
        binary (bool, opcional): Flag de mapeamento binário. Padrão é True.
    """
    # Garante que a pasta Result existe para não dar erro no f.open
    os.makedirs('Result', exist_ok=True) 
    
    I = len(Unitaries)
    J = len(Encodings)

    for i in range(I):
        for j in range(J):
            # 'a' abre o arquivo em modo Append (adiciona ao final sem apagar o que existe)
            f = open('Result/result.txt', 'a')
            U = Unitaries[i]
            U_params = U_num_params[i]
            Encoding = Encodings[j]
            Embedding = Encoding_to_Embedding(Encoding)

            # 1. Carrega e pré-processa os dados
            X_train, X_test, Y_train, Y_test = data.data_load_and_process(
                dataset, classes=classes, feature_reduction=Encoding, binary=binary
            )

            print("\n")
            print(f"Loss History for {circuit} circuits, {U} {Encoding} with {cost_fn}")
            
            # 2. Treina o circuito
            loss_history, trained_params = Training.circuit_training(
                X_train, Y_train, U, U_params, Embedding, circuit, cost_fn
            )

            # 3. Faz inferência no conjunto de Teste
            if circuit == 'QCNN':
                predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding, cost_fn) for x in X_test]
            elif circuit == 'Hierarchical':
                predictions = [Hierarchical_circuit.Hierarchical_classifier(x, trained_params, U, U_params, Embedding, cost_fn) for x in X_test]

            # 4. Avalia a precisão
            accuracy = accuracy_test(predictions, Y_test, cost_fn, binary)
            print(f"Accuracy for {U} {Encoding} : {accuracy}")

            # 5. Salva os logs
            f.write(f"Loss History for {circuit} circuits, {U} {Encoding} with {cost_fn}\n")
            f.write(str(loss_history) + "\n")
            f.write(f"Accuracy for {U} {Encoding} : {accuracy}\n\n")
            f.close()


def Data_norm(dataset: str, classes: List[int], Encodings: List[str], binary: bool = True) -> None:
    """
    Função de diagnóstico para analisar o comportamento da norma (L2) dos dados 
    após a redução de dimensionalidade clássica.

    Sorteia 10.000 amostras e particiona os vetores exatamente da mesma forma 
    que as funções 'Hybrid' fariam. Avalia a média e o desvio padrão da norma 
    desses sub-vetores para entender se os dados tendem a estourar a normalização.

    Args:
        dataset (str): Nome do dataset.
        classes (list): Classes a serem filtradas.
        Encodings (list de str): Métodos de extração (focado em pca32 e pca16).
        binary (bool): Flag de mapeamento.
    """
    os.makedirs('Result', exist_ok=True)
    J = len(Encodings)
    Num_data = 10000

    f = open('Result/data_norm.txt', 'a')

    for j in range(J):
        Encoding = Encodings[j]

        X_train, _, _, _ = data.data_load_and_process(
            dataset, classes=classes, feature_reduction=Encoding, binary=binary
        )

        if Encoding in ['pca32-3', 'autoencoder32-3']:
            norms_X1, norms_X2 = [], []
            for i in range(Num_data):
                index = np.random.randint(0, len(X_train))
                X = X_train[index]

                # Fatiamento e cálculo da norma
                X1 = X[:16] # 2**4
                X2 = X[16:32] # 2**4 : 2**5
                norms_X1.append(np.linalg.norm(X1))
                norms_X2.append(np.linalg.norm(X2))

            mean_X1, stdev_X1 = np.mean(norms_X1), np.std(norms_X1)
            mean_X2, stdev_X2 = np.mean(norms_X2), np.std(norms_X2)

            f.write(f"{Encoding} Encoding\n")
            f.write(f"mean of X1: {mean_X1} standard deviation of X1: {stdev_X1}\n")
            f.write(f"mean of X2: {mean_X2} standard deviation of X2: {stdev_X2}\n")

        elif Encoding in ['pca16', 'autoencoder16']:
            norms_X1, norms_X2, norms_X3, norms_X4 = [], [], [], []
            for i in range(Num_data):
                index = np.random.randint(0, len(X_train))
                X = X_train[index]

                X1, X2, X3, X4 = X[:4], X[4:8], X[8:12], X[12:16]
                norms_X1.append(np.linalg.norm(X1))
                norms_X2.append(np.linalg.norm(X2))
                norms_X3.append(np.linalg.norm(X3))
                norms_X4.append(np.linalg.norm(X4))

            mean_X1, stdev_X1 = np.mean(norms_X1), np.std(norms_X1)
            mean_X2, stdev_X2 = np.mean(norms_X2), np.std(norms_X2)
            mean_X3, stdev_X3 = np.mean(norms_X3), np.std(norms_X3)
            mean_X4, stdev_X4 = np.mean(norms_X4), np.std(norms_X4)

            f.write(f"{Encoding} Encoding\n")
            f.write(f"mean of X1: {mean_X1} standard deviation of X1: {stdev_X1}\n")
            f.write(f"mean of X2: {mean_X2} standard deviation of X2: {stdev_X2}\n")
            f.write(f"mean of X3: {mean_X3} standard deviation of X3: {stdev_X3}\n")
            f.write(f"mean of X4: {mean_X4} standard deviation of X4: {stdev_X4}\n")

    f.close()