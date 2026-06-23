# Advanced For-Loop for QML algorithm search
- 作者: FuTe Wong
- 提交年份: 2025年6月23日（v1）
- 出处: arXiv preprint, arXiv:2506.18260v1, 分类 cs.AI

## 研究内容
基于大型语言模型的多代理系统(LLMMA)来自动搜索和优化量子机器学习(QML)算法。

## LLMMA QML算法搜索方式
给定经典深度学习算法(如前向算法或反向传播算法)，系统可以促进开发工作流程找到其优化的量子对等体。
![](/blogs/algorithm-search/cb66e213ad066503.png)
<div align="center"> QML搜索流程流程图</div>

实验以三种经典算法为输入，测试系统的量子转换能力：                                                                                                                                                                                                                                                        
1.	多层感知机Multi-Layer Perceptron (MLP) → Quantum MLP：展示模型确实在学习（训练动态为正）
```python
# initialize the network
def initialize_network(n_inputs, hidden_layers, n_outputs):
    network = []
    # Create hidden layers
    for i in range(len(hidden_layers)):
        if i == 0:
            # First hidden layer
            layer = [{'weights': [random() for _ in range(n_inputs + 1)]}
                     for _ in range(hidden_layers[i])]
        else:
            # Subsequent hidden layers
            layer = [{'weights': [random() for _ in range(hidden_layers[i-1] + 1)]}
                     for _ in range(hidden_layers[i])]
        network.append(layer)

    # Output layer
    output_layer = [{'weights': [random() for _ in range(hidden_layers[-1] + 1)]}
                    for _ in range(n_outputs)]
    network.append(output_layer)
    return network


def forward_propagate(network, inputs):
    current_inputs = inputs

    for layer in network:
        new_inputs = []
        for neuron in layer:
            # Calculate weighted sum
            activation = neuron['weights'][-1]  # Bias
            for i in range(len(neuron['weights'])-1):
                activation += neuron['weights'][i] * current_inputs[i]
```

```python
def quantum_forward(params, input_data):
    """Execute quantum circuit with given parameters and input data."""
    backend = Aer.get_backend('qasm_simulator')

    # Create circuit
    qc, circuit_params = create_quantum_circuit(n_qubits, n_layers)

    # Assign parameters
    parameter_dict = dict(zip(circuit_params, params))
    bound_circuit = qc.assign_parameters(parameter_dict)

    # Encode input
    bound_circuit = encode_input(input_data.flatten()[:n_qubits], bound_circuit, n_qubits)

    # Execute
    job = backend.run(bound_circuit, shots=n_shots)
    result = job.result()
    counts = result.get_counts()

    # Calculate expectation value
    expectation = 0
    total_shots = sum(counts.values())

    for bitstring, count in counts.items():
        # Convert bitstring to +1/-1 value
        value = (-1)**(bitstring.count('1') % 2)
        expectation += value * count / total_shots

    return expectation


def cost_function(params, input_data, target):
    """Calculate cost function for optimization with L2 regularization."""
    prediction = quantum_forward(params, input_data)
```
2.前向算法Forward-Forward Algorithm（Hinton, 2022）→ Quantum Forward-Forward (QFF)：正/负通道被量化为量子电路
```python
# The forward-forward network
class ForwardForwardLayer(nn.Module):
    def __init__(self, input_size, output_size, threshold=2.0):
        super(ForwardForwardLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.threshold = threshold

    def forward(self, x):
        return self.linear(x)

    def compute_goodness(self, x):
        """Compute the goodness score for the input"""
        h = self.linear(x)
        return torch.sum(h**2, dim=1)  # Sum of squared activations


class ForwardForwardNetwork(nn.Module):
    def __init__(self, layer_sizes, threshold=2.0, learning_rate=0.01):
        super(ForwardForwardNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.threshold = threshold
        self.learning_rate = learning_rate

        # Create layers
        for i in range(len(layer_sizes)-1):
            self.layers.append(ForwardForwardLayer(layer_sizes[i], layer_sizes[i+1], threshold))

    def forward_positive(self, x, layer_idx):
        """Forward pass for positive (real) samples"""
        if layer_idx == 0:
            return self.layers[layer_idx].compute_goodness(x)
```

```python
class QuantumForwardForward:
    def compute_goodness(self, x):
        """Compute quantum goodness score using measurement statistics"""
        qc = self.create_quantum_circuit(x)
        qc.measure_all()

        # Run circuit and get counts
        job = self.backend.run(qc, shots=n_shots)
        result = job.result()
        counts = result.get_counts()

        # Calculate weighted sum of measurement outcomes
        total_weight = 0
        for bitstring, count in counts.items():
            # Weight each measurement by number of 1s (Hamming weight)
            weight = sum(int(bit) for bit in bitstring)
            total_weight += weight * count

        # Normalize by total shots and number of qubits
        normalized_weight = total_weight / (n_shots * self.n_qubits)
        # Apply tanh activation with increased scaling for better separation
        goodness = np.tanh(4.0 * normalized_weight - 2.0)

        return goodness

    def cost_function(self, parameters, x_pos, x_neg):
        """Compute cost function for positive and negative samples"""
        self.parameters = parameters

        # Compute goodness scores
        pos_goodness = self.compute_goodness(x_pos)
        neg_goodness = self.compute_goodness(x_neg)

        # Cost is higher when positive samples have high goodness
        # and negative samples have low goodness
        cost = -pos_goodness + neg_goodness
        return cost
```

3. 反向传播算法Backpropagation Algorithm → Quantum Backpropagation (QBP)：梯度反向传播被替换为 parameter-shift rule1.多层感知机MLP

```python
# Backpropagation algorithm
def backpropagation(network, X, y, learning_rate):
    """
    Implements the backpropagation algorithm for a neural network

    Parameters:
    network: list of layers in the neural network
    X: input data
    y: target values
    learning_rate: learning rate for weight updates
    """

    # Forward Pass
    def forward_pass(X):
        activations = [X]  # List to store activations for each layer
        for layer in network:
            # Calculate net input for current layer
            net = np.dot(activations[-1], layer['weights']) + layer['bias']
            # Apply activation function
            activation = sigmoid(net)
            activations.append(activation)
        return activations

    # Sigmoid activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Derivative of sigmoid function
    def sigmoid_derivative(x):
        return x * (1 - x)

    # Forward propagation
    activations = forward_pass(X)
```

```python
def quantum_gradient(params, x, y_true):
    """Calculate gradient using parallel parameter shift."""
    epsilon = 0.01
    grads = np.zeros_like(params)

    # Calculate base cost once
    base_cost = cost_function(params, x, y_true)

    # Calculate shifted costs in parallel
    for i in range(0, len(params), 2):  # Process parameters in pairs
        batch_params = []
        indices = []

        # Create parameter shifts for current batch
        for j in range(2):
            if i + j < len(params):
                params_plus = params.copy()
                params_plus[i + j] += epsilon
                batch_params.append(params_plus)

                params_minus = params.copy()
                params_minus[i + j] -= epsilon
                batch_params.append(params_minus)
                indices.append(i + j)

        # Calculate costs for all shifts in current batch
        batch_costs = [cost_function(p, x, y_true) for p in batch_params]

        # Process results
        for idx, j in enumerate(indices):
            grads[j] = (batch_costs[idx*2] - batch_costs[idx*2 + 1]) / (2 * epsilon)

    return grads
```

性能评估：                                                                                              
  基线：基线人工制作的后变分量子神经网络                      
  数据集：UCI ML手写数字数据集                 
  基础LLM：claude-3-5-sonnet-20240620                                                                  
  指标：测试集平均准确率
| 模型 | 测试集平均准确率（%） |
| ---- | -------------------- |
| Baseline（QNN，Pennylane手工） | 15.55 |
| QMLP（量子MLP） | 9.40 |
| QFF（量子 Forward-Forward） | 15.17 |
| QBP（量子 Backpropagation） | 12.37 |

讨论：代理框架用于搜索量子机器学习算法，目标是比经典同行更高效和有效。此文工作侧重于将经典机器学习算法初始翻译为其量子对应物。工作的未来方向将是研究如何有效地搜索经典机器学习概念的空间，并将其转化为量子机器学习算法。