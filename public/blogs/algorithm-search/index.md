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