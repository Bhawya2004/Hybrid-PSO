from flask import Flask, request, jsonify, render_template
import numpy as np
from scipy.special import expit  # sigmoid function
import json
import numexpr as ne

app = Flask(__name__)

class NeuralNetwork:
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights = None
        self.biases = None
        
    def set_parameters(self, params):
        # Extract weights and biases from params
        w1_size = self.input_size * self.hidden_size
        w2_size = self.hidden_size * self.output_size
        b1_size = self.hidden_size
        b2_size = self.output_size
        
        w1 = params[:w1_size].reshape(self.input_size, self.hidden_size)
        w2 = params[w1_size:w1_size+w2_size].reshape(self.hidden_size, self.output_size)
        b1 = params[w1_size+w2_size:w1_size+w2_size+b1_size]
        b2 = params[w1_size+w2_size+b1_size:]
        
        self.weights = [w1, w2]
        self.biases = [b1, b2]
        
    def forward(self, X):
        # Forward pass
        z1 = np.dot(X, self.weights[0]) + self.biases[0]
        a1 = expit(z1)  # sigmoid activation
        z2 = np.dot(a1, self.weights[1]) + self.biases[1]
        return z2  # linear output for regression
    
    def get_param_count(self):
        # Calculate total number of parameters
        w1_size = self.input_size * self.hidden_size
        w2_size = self.hidden_size * self.output_size
        b1_size = self.hidden_size
        b2_size = self.output_size
        return w1_size + w2_size + b1_size + b2_size

class PSO:
    def __init__(self, n_particles, dimensions, c1=1.5, c2=1.5, w=0.7, bounds=(-5, 5)):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.w = w    # inertia weight
        self.bounds = bounds
        
        # Initialize particles and velocities
        self.particles = np.random.uniform(bounds[0], bounds[1], (n_particles, dimensions))
        self.velocities = np.random.uniform(-1, 1, (n_particles, dimensions))
        
        # Initialize personal best and global best
        self.pbest = self.particles.copy()
        self.pbest_fitness = np.full(n_particles, float('inf'))
        self.gbest = np.zeros(dimensions)
        self.gbest_fitness = float('inf')
        
    def update(self, fitness_func):
        # Evaluate fitness for each particle
        for i in range(self.n_particles):
            fitness = fitness_func(self.particles[i])
            
            # Update personal best
            if fitness < self.pbest_fitness[i]:
                self.pbest[i] = self.particles[i].copy()
                self.pbest_fitness[i] = fitness
                
                # Update global best
                if fitness < self.gbest_fitness:
                    self.gbest = self.particles[i].copy()
                    self.gbest_fitness = fitness
        
        # Update velocities and positions
        r1 = np.random.random((self.n_particles, self.dimensions))
        r2 = np.random.random((self.n_particles, self.dimensions))
        
        self.velocities = (self.w * self.velocities + 
                          self.c1 * r1 * (self.pbest - self.particles) + 
                          self.c2 * r2 * (self.gbest - self.particles))
        
        self.particles += self.velocities
        
        # Apply bounds
        self.particles = np.clip(self.particles, self.bounds[0], self.bounds[1])
        
    def optimize(self, fitness_func, iterations):
        for _ in range(iterations):
            self.update(fitness_func)
        return self.gbest, self.gbest_fitness

def evaluate_function(func_str, x_values):
    # Safely evaluate the function for each x value
    x = x_values  # numexpr uses 'x' as the variable
    try:
        return ne.evaluate(func_str)
    except Exception as e:
        raise ValueError(f"Error evaluating function: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/approximate', methods=['POST'])
def approximate():
    data = request.get_json()
    
    # Extract parameters from request
    func_str = data.get('function', 'sin(x)')
    x_min = float(data.get('x_min', -5))
    x_max = float(data.get('x_max', 5))
    n_points = int(data.get('n_points', 100))
    
    # PSO parameters
    n_particles = int(data.get('n_particles', 30))
    iterations = int(data.get('iterations', 100))
    c1 = float(data.get('c1', 1.5))
    c2 = float(data.get('c2', 1.5))
    w = float(data.get('w', 0.7))
    hidden_size = int(data.get('hidden_size', 10))
    
    # Generate x values and evaluate true function
    x_values = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
    try:
        y_true = evaluate_function(func_str, x_values.flatten())
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    # Initialize neural network
    nn = NeuralNetwork(input_size=1, hidden_size=hidden_size, output_size=1)
    param_count = nn.get_param_count()
    
    # Define fitness function for PSO
    def fitness_func(params):
        nn.set_parameters(params)
        y_pred = nn.forward(x_values).flatten()
        return np.mean((y_pred - y_true) ** 2)  # MSE
    
    # Run PSO optimization
    pso = PSO(n_particles=n_particles, dimensions=param_count, 
              c1=c1, c2=c2, w=w, bounds=(-5, 5))
    best_params, best_fitness = pso.optimize(fitness_func, iterations)
    
    # Get final predictions
    nn.set_parameters(best_params)
    y_pred = nn.forward(x_values).flatten()
    
    # Calculate R² score
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2_score = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    # Prepare response
    result = {
        'x_values': x_values.flatten().tolist(),
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'mse': float(best_fitness),
        'r2_score': float(r2_score)
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)