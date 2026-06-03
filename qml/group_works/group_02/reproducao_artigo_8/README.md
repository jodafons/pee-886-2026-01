# Repdodução dos experimentos de QCNN

Este repositório foi derivado [daqui](https://github.com/takh04/QCNN).

## Explicação sobre o Contexto de Enfileiramento do PennyLane

Em Python clássico, nós estamos acostumados com um fluxo de dados funcional: você passa uma variável para uma função, ela sofre uma transformação e a função `return` a nova variável. No PennyLane, a construção de circuitos funciona baseada no que chamamos de "Contexto de Enfileiramento" (Queuing Context) ou, em termos mais simples, um gravador de fita. 

Aqui está o que acontece por baixo dos panos: quando você decora uma função com `@qml.qnode(dev)`, você está transformando aquela função Python comum em um "Nó Quântico". O decorador diz para o PennyLane: *"A partir do momento que esta função for chamada, ligue o gravador e preste atenção em todas as operações quânticas que forem instanciadas, porque elas devem ser executadas no **dispositivo dev**."*


Exemplo:
```python
import pennylane as qml

dev = qml.device("default.qubit", wires=8)

@qml.qnode(dev)
def meu_circuito_qcnn(X, params):
    # 1. O gravador liga aqui (Contexto do QNode iniciado)
    
    # 2. Chamamos a função. Ela não retorna nada (None), mas lá dentro, 
    # cada porta RX, RY ou Möttönen que ela chama é adicionada à fita de gravação.
    data_embedding(X, embedding_type='Angle-compact') 
    
    # 3. Chamamos outras funções do circuito (as convoluções, pooling, etc)
    QCNN_circuit(params) 
    
    # 4. O gravador para e devolve o valor esperado (medida)
    return qml.expval(qml.PauliZ(0))
```

Graças a esse comportamento in-place invisível do @qml.qnode, o autor pôde isolar a lógica chata de inicialização dos dados no arquivo data.py (na função data_embedding), mantendo o script principal limpo e legível. A função simplesmente atua sobre o espaço de qubits (os wires) que já foi alocado no nível superior pelo QNode.

## Códigos auxiliares

### QCNN/data.py

contém função que baixa dataset (MNIST ou Fashion MNIST) via Keras e o carrega em memória, adiciona uma dimensão nas imagens (28x28 para 28x28x1), normaliza os pixels, transforma o problema em binário (mantém duas classes), achata o vetor (2D para 1D), reduz a dimensionalidade (por redimensionamento, PCA ou autoencoder), transforma em ângulo entre 0 e pi e retorna X_train (Num samples, Dimension), X_test, Y_train (Num samples), Y_test.

### QCNN/embedding.py

contém função que recebe um vetor 1D de features de uma das amostras (X) e o tipo de embedding. Ela vai ser chamada dentro de um contexto `@qml.qnode(dev)`, na linha 80 de `QCNN_circuit.py`. Essa linha por sua vez é chamada tanto no treinamento (linha 155 de `Benchmarking.py`) quanto na inferência (linha 158 `Benchmarking.py`). Essa chamada dentro de um decorador `@qml.qnode(dev)` garante que as funções de embedding com a amostra atuem sobre os qubits definidos para o device `dev`. Com o `AmplitudeEmbedding`, 8 qubits representam 2^8 = 256 números reais, onde cada um deles é a probabilidade de um dos estados, de `|00000000>` a `|11111111>`, permitindo a codificação de um vetor de 256 números em 8 qubits. Com o `AngleEmbedding`, um vetor de 8 números vira 8 ângulos de rotação de qubits em torno do Y. Com o `AngleEmbedding`, cada **par** de elmentos de um vetor de 16 números vira um ângulo de rotação de um qubit `|00000000>` em torno de X e um ângulo de rotação em torno de Y, totalizando 8 qubits. O `MottonenStatePreparation` é um tipo de embedding de amplitude utilizado para fazer Hybrid Direct Embedding (HDE), no qual um vetor de 32 números é quebrado em dois blocos de 16 features na estratégia `Amplitude-Hybrid4-i`, ou quatro blocos de 4 features na `Amplitude-Hybrid2-i`, e, em seguida, escolhe quais qubits cada número vai codificar (o algoritmo original de Möttönen pega as amplitudes clássicas e faz uma trigonometria reversa pesada para descobrir quais devem ser os ângulos de rotação no circuito para gerar aquele estado).  Por exemplo, o `Amplitude-Hybrid2-1` faz elementos 1 a 4 serem codificados pelos qubits 0/1 (elementos 12 a 16 serem codificados pelos qubits 6/7), enquanto que o `Amplitude-Hybrid2-1` faz elementos 1 a 4 serem codificados pelos qubits 0/4 (elementos 12 a 16 serem codificados pelos qubits 3/7), criando um stride naas convoluções, uma vez que uma porta que processa 2 qubits vizinhos vai combinar features afastadas. Veja o item abaixo para ver mais sobre encodings híibridos.

### QCNN/Angular_hybrid.py

implementa algumas funções usadas no `QCNN/embedding.py` para possibilitar Hybrid Angle Embedding (HAE) pegando as features e injetando diretamente como os ângulos de rotação em Y controlados por outras features. 

### QCNN/unitary.py 

implementações das sequências de portas (ansatz) utilizadas nas camadas convolucionais. Unitários básicos (U 5,6,9,13,14,15), tensor tree networks (U TTN) e grupos especiais (U SO4,SU4). Implementação das portas utilziadas nas camadas de pooling.

### QCNN/QCNN_circuit.py 

a função `QCNN_structure` implementa uma QCNN com três aplicações alternadas de convolução e pooling, recebendo o vetor de parâmetros e o distribuindo entre as seis camadas. As camadas criadas internamente pela função `QCNN_structure` são importadas de `QCNN/unitary.py`. Para cirar as camadas convolucionais, deve-se passar como primeiro argumento uma porta U, que implementa o ansatz usado como base para a convolução. O código também implementa variações, como uma QCNN sem pooling e uma QCNN 1D. Este código contém o coração do repositório, uma vez que define o device e cria o QNode (classe `QCNN`, com decorador `@qml.qnode(dev)`), que inclui o embedding (entrada com 1 amostra X), a definição da rede e a medição (saída, usando MSE ou Cross Entropy).

### QCNN/Training.py

define os parâmetros globais de treinamento, como passos, learning rate e batch size. A função `circuit_training` recebe os dados (X e y) de treino, bem como informações sobre a rede (tipo de embedding, ansatz, número de parâmetros a serem inicializados aleatoriamente, função de custo usada), e implementa o loop de treinamento. Para cada iteração do loop, é selecionado de forma aleatória um mini-batch que é passado pela rede pra calcular o custo a ser otimizado. A passagem pela rede e o cálculo do custo ocorrem na chamda pela função `cost`. O otimizador `NesterovMomentumOptimizer` do PennyLane recebe a função a ser otimizada (`cost`) e a lista de parâmetros e usa algo como a Parameter-Shift Rule para autalizar a lista de parâmetros. Isso só é possível porque as funções de perda como a `cross_entropy` utilziam o `autograd` para diferenciação automática.

```python
# Dentro da função cost
predictions = [QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]
if cost_fn == 'cross_entropy':
    return cross_entropy(Y, predictions)

# dentro da função circuit_training
params = np.random.randn(total_params, requires_grad=True)    
opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
for it in range(steps):
    X_batch, Y_batch = get_random_batch(X_train, Y_train)
    lambda c: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn)
    params, _ = opt.step_and_cost(c, params)
...
return loss_history, trained_params
```

### QCNN/Benchmarking.py

executa grid search de unitários e embeddings (loop dentro de loop). Em cada iteração, abre arquivo para resultados no modo append (dentro da pasta `Result`, criada automaticamente caso não exista), carrega o dado com a técnica de encoding mais adequada pro embedding a ser utilziado, treina o circuito (`circuit_training`), carrega rede com parâmetros treinados recebidos (`QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding, cost_fn)`), passa cada amostra `x` do conjunto de testes pela rede (repete linha enterior `for x in X_test` para gerar lista de predições) e calcula a acurácia das predições. 

## Realizando experimentos 

Acesse o arquivo `QCNN/result.py` e ajuste os parâmetros a serem variados para teste

```python
Unitaries = ['U_SU4', 'U_SU4_1D', 'U_SU4_no_pooling', 'U_9_1D']
U_num_params = [15, 15, 15, 2]
Encodings = ['resize256']
dataset = 'fashion_mnist'
classes = [0,1]
binary = False
cost_fn = 'cross_entropy'
```

Execute:

```bash
cd QCNN
python result.py
```

Esse código irá executar o `Benchmarking` descrito acima