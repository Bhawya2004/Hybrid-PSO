document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const form = document.getElementById('approximation-form');
    const runBtn = document.getElementById('run-btn');
    const loading = document.getElementById('loading');
    const plotContainer = document.getElementById('plot-container');
    const metrics = document.getElementById('metrics');
    const mseValue = document.getElementById('mse-value');
    const r2Value = document.getElementById('r2-value');
    
    // Range input value displays
    const rangeInputs = [
        { input: 'n-points', display: 'n-points-value' },
        { input: 'hidden-size', display: 'hidden-size-value' },
        { input: 'n-particles', display: 'n-particles-value' },
        { input: 'iterations', display: 'iterations-value' },
        { input: 'c1', display: 'c1-value' },
        { input: 'c2', display: 'c2-value' },
        { input: 'w', display: 'w-value' }
    ];
    
    // Update range input displays
    rangeInputs.forEach(({ input, display }) => {
        const inputElement = document.getElementById(input);
        const displayElement = document.getElementById(display);
        
        // Initial value
        displayElement.textContent = inputElement.value;
        
        // Update on change
        inputElement.addEventListener('input', () => {
            displayElement.textContent = inputElement.value;
        });
    });
    
    // Initialize empty plot
    document.addEventListener('DOMContentLoaded', function() {
        // Make sure this code runs after the DOM is fully loaded
        const plotContainer = document.getElementById('plot-container');
        if (plotContainer) {
            Plotly.newPlot(plotContainer, [], {
                title: 'Function Approximation',
                xaxis: { title: 'x' },
                yaxis: { title: 'y' },
                margin: { t: 50, b: 50, l: 50, r: 50 }
            });
        } else {
            console.error('Plot container not found');
        }
    });
    
    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        runBtn.disabled = true;
        runBtn.classList.add('button-processing'); // Add animation class
        loading.classList.remove('hidden');
        metrics.classList.add('hidden'); // Hide previous metrics when starting new calculation
        
        // Get form values
        const functionStr = document.getElementById('function').value;
        const xMin = parseFloat(document.getElementById('x-min').value);
        const xMax = parseFloat(document.getElementById('x-max').value);
        const nPoints = parseInt(document.getElementById('n-points').value);
        const nParticles = parseInt(document.getElementById('n-particles').value);
        const iterations = parseInt(document.getElementById('iterations').value);
        const c1 = parseFloat(document.getElementById('c1').value);
        const c2 = parseFloat(document.getElementById('c2').value);
        const w = parseFloat(document.getElementById('w').value);
        const hiddenSize = parseInt(document.getElementById('hidden-size').value);
        
        // Prepare request data
        const requestData = {
            function: functionStr,
            x_min: xMin,
            x_max: xMax,
            n_points: nPoints,
            n_particles: nParticles,
            iterations: iterations,
            c1: c1,
            c2: c2,
            w: w,
            hidden_size: hiddenSize
        };
        
        try {
            // Send request to backend
            const response = await fetch('/approximate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to process request');
            }
            
            const data = await response.json();
            
            // Update plot
            const trueTrace = {
                x: data.x_values,
                y: data.y_true,
                type: 'scatter',
                mode: 'lines',
                name: 'True Function',
                line: { color: 'blue', width: 2 }
            };
            
            const predTrace = {
                x: data.x_values,
                y: data.y_pred,
                type: 'scatter',
                mode: 'markers',
                name: 'PSO-NN Approximation',
                marker: { color: 'red', size: 6 }
            };
            
            Plotly.react(plotContainer, [trueTrace, predTrace], {
                title: 'Function Approximation',
                xaxis: { title: 'x' },
                yaxis: { title: 'y' },
                margin: { t: 50, b: 50, l: 50, r: 50 }
            });
            
            // Update metrics
            mseValue.textContent = data.mse.toExponential(4);
            r2Value.textContent = data.r2_score.toFixed(4);
            metrics.classList.remove('hidden');
            
        } catch (error) {
            alert('Error: ' + error.message);
            console.error('Error:', error);
        } finally {
            // Reset loading state
            runBtn.disabled = false;
            runBtn.classList.remove('button-processing'); // Remove animation class
            loading.classList.add('hidden');
        }
    });
});