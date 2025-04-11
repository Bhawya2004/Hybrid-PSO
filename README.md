# PSO-Neural Network Function Approximation

A web application that approximates complex mathematical functions using a hybrid Particle Swarm Optimization (PSO) and Neural Network approach.

## Overview

This application allows users to input mathematical functions and see how well a neural network, optimized using PSO, can approximate them. The system visualizes both the true function and the approximated function, providing metrics to evaluate the quality of the approximation.

## Features

- **Function Input**: Enter any mathematical function using standard notation
- **Parameter Configuration**: Adjust x-range, number of data points, and PSO parameters
- **Neural Network Configuration**: Set the size of the hidden layer
- **Interactive Visualization**: Compare the true function with the PSO-NN approximation
- **Performance Metrics**: View Mean Squared Error (MSE) and R² score

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript, Plotly.js
- **Backend**: Python, Flask
- **Algorithms**: Neural Networks, Particle Swarm Optimization

## How It Works

1. **Neural Network Architecture**:
   - Input layer: 1 neuron (x value)
   - Hidden layer: Configurable number of neurons with sigmoid activation
   - Output layer: 1 neuron (y value) with linear activation

2. **Particle Swarm Optimization**:
   - Each particle represents a set of weights and biases for the neural network
   - Particles move through the parameter space to minimize the mean squared error
   - Parameters like cognitive coefficient, social coefficient, and inertia weight control particle behavior

3. **Approximation Process**:
   - User inputs a function and parameters
   - Backend generates data points from the true function
   - PSO optimizes neural network weights to minimize error
   - Results are sent back to frontend for visualization

## Installation

1. Clone the repository
2. Install required Python packages: