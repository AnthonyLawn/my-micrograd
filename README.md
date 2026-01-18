# my-micrograd

My recreation of the micrograd engine by Andrej Karpathy.

## Overview

This project implements a minimal automatic differentiation engine and a simple neural network library, inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy. It is written from scratch for educational purposes.

## Features

- Scalar-valued automatic differentiation (`Value` class)
- Operator overloading for arithmetic and activation functions
- Backpropagation for gradient computation
- Simple neural network modules: Neuron, Layer, and MLP
- Example usage in a Jupyter notebook for regression on Call of Duty match data

## Project Structure

- [`my_micrograd/engine.py`](my_micrograd/engine.py): Core autodiff engine (`Value` class)
- [`my_micrograd/nn.py`](my_micrograd/nn.py): Neural network modules (MLP, Layer, Neuron)
- [`tests/test_engine.py`](tests/test_engine.py): Unit tests comparing this engine to the original micrograd
- [`call_of_duty_analysis.ipynb`](call_of_duty_analysis.ipynb): Example notebook using the engine and MLP on real data
- [`data/call_of_duty_matches.csv`](data/call_of_duty_matches.csv): Example dataset

## Installation

Clone the repository and install the required dependencies (e.g., `pandas`, `numpy`, `scikit-learn`, `matplotlib`).

```sh
pip install -r requirements.txt
```

## Usage Example

```python
from my_micrograd.engine import Value
from my_micrograd.nn import MLP

# Create a simple MLP with 2 inputs and one output
model = MLP(2, [4, 4, 1])

# Example input
x = [Value(0.5), Value(0.8)]
pred = model(x)
print(pred)
```

See [`call_of_duty_analysis.ipynb`](call_of_duty_analysis.ipynb) for a full example of training and evaluating a neural network on real data.

## Testing

Run all tests with:

```sh
pytest
```

## License

This project is for educational purposes only.
