import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Play, Pause, RotateCcw, Settings, BookOpen, Brain } from 'lucide-react';

const NeuralNetworkVisualizer = () => {
  const svgRef = useRef();
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [mode, setMode] = useState('forward'); // 'forward', 'backward', 'training'
  const [networkData, setNetworkData] = useState(null);
  const [trainingData, setTrainingData] = useState({ input: [0.3, 0.9], target: [1.0] });
  const [learningRate, setLearningRate] = useState(0.1);
  const [showHints, setShowHints] = useState(true);
  const [activationFunction, setActivationFunction] = useState('sigmoid');
  
  // Network architecture: [input, hidden, output]
  const [architecture, setArchitecture] = useState([2, 3, 1]);
  const [weights, setWeights] = useState([]);
  const [biases, setBiases] = useState([]);
  const [activations, setActivations] = useState([]);
  const [gradients, setGradients] = useState([]);
  const [loss, setLoss] = useState(0);

  // Initialize network with random weights
  useEffect(() => {
    initializeNetwork();
  }, [architecture]);

  const initializeNetwork = () => {
    const newWeights = [];
    const newBiases = [];
    const newActivations = [];
    
    // Initialize weights and biases randomly
    for (let i = 0; i < architecture.length - 1; i++) {
      const weightMatrix = [];
      for (let j = 0; j < architecture[i]; j++) {
        const row = [];
        for (let k = 0; k < architecture[i + 1]; k++) {
          row.push((Math.random() - 0.5) * 2);
        }
        weightMatrix.push(row);
      }
      newWeights.push(weightMatrix);
      
      const biasArray = [];
      for (let j = 0; j < architecture[i + 1]; j++) {
        biasArray.push((Math.random() - 0.5) * 0.5);
      }
      newBiases.push(biasArray);
    }
    
    // Initialize activations
    architecture.forEach((size, i) => {
      newActivations.push(new Array(size).fill(0));
    });
    
    setWeights(newWeights);
    setBiases(newBiases);
    setActivations(newActivations);
    setCurrentStep(0);
    
    // Set input values
    const newActivations2 = [...newActivations];
    newActivations2[0] = [...trainingData.input];
    setActivations(newActivations2);
    
    drawNetwork();
  };

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const sigmoidDerivative = (x) => x * (1 - x);
  const relu = (x) => Math.max(0, x);
  const reluDerivative = (x) => x > 0 ? 1 : 0;

  const applyActivation = (x, func = activationFunction) => {
    switch (func) {
      case 'sigmoid': return sigmoid(x);
      case 'relu': return relu(x);
      case 'tanh': return Math.tanh(x);
      default: return x;
    }
  };

  const applyActivationDerivative = (x, func = activationFunction) => {
    switch (func) {
      case 'sigmoid': return sigmoidDerivative(x);
      case 'relu': return reluDerivative(x);
      case 'tanh': return 1 - x * x;
      default: return 1;
    }
  };

  const forwardPass = () => {
    const newActivations = [...activations];
    newActivations[0] = [...trainingData.input];
    
    for (let layer = 1; layer < architecture.length; layer++) {
      for (let neuron = 0; neuron < architecture[layer]; neuron++) {
        let sum = biases[layer - 1][neuron];
        for (let prevNeuron = 0; prevNeuron < architecture[layer - 1]; prevNeuron++) {
          sum += newActivations[layer - 1][prevNeuron] * weights[layer - 1][prevNeuron][neuron];
        }
        newActivations[layer][neuron] = applyActivation(sum);
      }
    }
    
    setActivations(newActivations);
    
    // Calculate loss (MSE)
    const output = newActivations[newActivations.length - 1];
    const target = trainingData.target;
    const newLoss = target.reduce((sum, t, i) => sum + Math.pow(t - output[i], 2), 0) / target.length;
    setLoss(newLoss);
    
    drawNetwork();
  };

  const backwardPass = () => {
    const newGradients = [];
    const outputLayer = activations.length - 1;
    const output = activations[outputLayer];
    const target = trainingData.target;
    
    // Initialize gradients for each layer
    for (let i = 0; i < activations.length; i++) {
      newGradients.push(new Array(activations[i].length).fill(0));
    }
    
    // Calculate output layer gradients (dL/dO)
    for (let i = 0; i < output.length; i++) {
      newGradients[outputLayer][i] = 2 * (output[i] - target[i]);
    }
    
    // Backpropagate through hidden layers
    for (let layer = outputLayer - 1; layer >= 0; layer--) {
      for (let neuron = 0; neuron < architecture[layer]; neuron++) {
        let gradient = 0;
        for (let nextNeuron = 0; nextNeuron < architecture[layer + 1]; nextNeuron++) {
          const activationDerivative = applyActivationDerivative(activations[layer + 1][nextNeuron]);
          gradient += newGradients[layer + 1][nextNeuron] * activationDerivative * weights[layer][neuron][nextNeuron];
        }
        newGradients[layer][neuron] = gradient;
      }
    }
    
    setGradients(newGradients);
    drawNetwork();
  };

  const updateWeights = () => {
    const newWeights = [...weights];
    const newBiases = [...biases];
    
    for (let layer = 0; layer < weights.length; layer++) {
      for (let i = 0; i < weights[layer].length; i++) {
        for (let j = 0; j < weights[layer][i].length; j++) {
          const gradient = gradients[layer + 1][j] * applyActivationDerivative(activations[layer + 1][j]) * activations[layer][i];
          newWeights[layer][i][j] -= learningRate * gradient;
        }
      }
      
      for (let j = 0; j < biases[layer].length; j++) {
        const gradient = gradients[layer + 1][j] * applyActivationDerivative(activations[layer + 1][j]);
        newBiases[layer][j] -= learningRate * gradient;
      }
    }
    
    setWeights(newWeights);
    setBiases(newBiases);
    drawNetwork();
  };

  const stepForward = () => {
    if (mode === 'forward') {
      if (currentStep === 0) {
        forwardPass();
        setCurrentStep(1);
      }
    } else if (mode === 'backward') {
      if (currentStep === 0) {
        forwardPass();
        setCurrentStep(1);
      } else if (currentStep === 1) {
        backwardPass();
        setCurrentStep(2);
      } else if (currentStep === 2) {
        updateWeights();
        setCurrentStep(0);
      }
    }
  };

  const reset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
    setGradients([]);
    initializeNetwork();
  };

  const drawNetwork = () => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    
    const width = 800;
    const height = 500;
    const layerWidth = width / architecture.length;
    
    svg.attr("width", width).attr("height", height);
    
    const nodePositions = [];
    
    // Calculate node positions
    architecture.forEach((layerSize, layerIndex) => {
      const layerPositions = [];
      const startY = (height - (layerSize - 1) * 60) / 2;
      
      for (let nodeIndex = 0; nodeIndex < layerSize; nodeIndex++) {
        layerPositions.push({
          x: layerIndex * layerWidth + layerWidth / 2,
          y: startY + nodeIndex * 60,
          layer: layerIndex,
          index: nodeIndex
        });
      }
      nodePositions.push(layerPositions);
    });
    
    // Draw connections (weights)
    for (let layer = 0; layer < weights.length; layer++) {
      for (let i = 0; i < weights[layer].length; i++) {
        for (let j = 0; j < weights[layer][i].length; j++) {
          const startPos = nodePositions[layer][i];
          const endPos = nodePositions[layer + 1][j];
          const weight = weights[layer][i][j];
          
          // Color based on weight value and gradient flow
          let strokeColor = "#666";
          let strokeWidth = Math.abs(weight) * 3 + 1;
          
          if (mode === 'backward' && currentStep >= 1 && gradients.length > 0) {
            const gradientMagnitude = Math.abs(gradients[layer + 1] ? gradients[layer + 1][j] : 0);
            strokeColor = d3.interpolateRdYlBu(1 - Math.min(gradientMagnitude, 1));
            strokeWidth = Math.max(strokeWidth, gradientMagnitude * 5 + 1);
          }
          
          svg.append("line")
            .attr("x1", startPos.x)
            .attr("y1", startPos.y)
            .attr("x2", endPos.x)
            .attr("y2", endPos.y)
            .attr("stroke", strokeColor)
            .attr("stroke-width", strokeWidth)
            .attr("opacity", 0.7);
          
          // Weight labels
          svg.append("text")
            .attr("x", (startPos.x + endPos.x) / 2)
            .attr("y", (startPos.y + endPos.y) / 2 - 5)
            .text(weight.toFixed(2))
            .attr("font-size", "10px")
            .attr("fill", "#333")
            .attr("text-anchor", "middle");
        }
      }
    }
    
    // Draw nodes
    nodePositions.forEach((layer, layerIndex) => {
      layer.forEach((pos, nodeIndex) => {
        const activation = activations[layerIndex] ? activations[layerIndex][nodeIndex] : 0;
        const gradient = gradients[layerIndex] ? gradients[layerIndex][nodeIndex] : 0;
        
        // Node color based on activation value
        let fillColor = d3.interpolateBlues(Math.abs(activation));
        if (layerIndex === 0) fillColor = "#4CAF50"; // Input layer
        if (layerIndex === architecture.length - 1) fillColor = "#FF9800"; // Output layer
        
        // Gradient glow effect during backprop
        if (mode === 'backward' && currentStep >= 1 && Math.abs(gradient) > 0.01) {
          svg.append("circle")
            .attr("cx", pos.x)
            .attr("cy", pos.y)
            .attr("r", 25 + Math.abs(gradient) * 10)
            .attr("fill", "none")
            .attr("stroke", "#FF5722")
            .attr("stroke-width", 2)
            .attr("opacity", 0.5);
        }
        
        // Main node circle
        svg.append("circle")
          .attr("cx", pos.x)
          .attr("cy", pos.y)
          .attr("r", 20)
          .attr("fill", fillColor)
          .attr("stroke", "#333")
          .attr("stroke-width", 2);
        
        // Activation value text
        svg.append("text")
          .attr("x", pos.x)
          .attr("y", pos.y + 3)
          .text(activation.toFixed(2))
          .attr("font-size", "12px")
          .attr("fill", "white")
          .attr("font-weight", "bold")
          .attr("text-anchor", "middle");
        
        // Gradient value (during backprop)
        if (mode === 'backward' && currentStep >= 1 && Math.abs(gradient) > 0.01) {
          svg.append("text")
            .attr("x", pos.x)
            .attr("y", pos.y - 30)
            .text(`∇: ${gradient.toFixed(3)}`)
            .attr("font-size", "10px")
            .attr("fill", "#FF5722")
            .attr("font-weight", "bold")
            .attr("text-anchor", "middle");
        }
        
        // Layer labels
        if (nodeIndex === 0) {
          let layerName = "";
          if (layerIndex === 0) layerName = "Input";
          else if (layerIndex === architecture.length - 1) layerName = "Output";
          else layerName = `Hidden ${layerIndex}`;
          
          svg.append("text")
            .attr("x", pos.x)
            .attr("y", layer[0].y - 50)
            .text(layerName)
            .attr("font-size", "14px")
            .attr("font-weight", "bold")
            .attr("fill", "#333")
            .attr("text-anchor", "middle");
        }
      });
    });
    
    // Add activation function label
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 30)
      .text(`Activation: ${activationFunction}`)
      .attr("font-size", "16px")
      .attr("font-weight", "bold")
      .attr("fill", "#333")
      .attr("text-anchor", "middle");
    
    // Add loss display
    svg.append("text")
      .attr("x", width - 100)
      .attr("y", 30)
      .text(`Loss: ${loss.toFixed(4)}`)
      .attr("font-size", "14px")
      .attr("font-weight", "bold")
      .attr("fill", "#E91E63")
      .attr("text-anchor", "middle");
  };

  useEffect(() => {
    if (weights.length > 0) {
      drawNetwork();
    }
  }, [weights, biases, activations, gradients, mode, currentStep]);

  const getCurrentStepDescription = () => {
    if (mode === 'forward') {
      return currentStep === 0 ? 'Ready for forward pass' : 'Forward pass completed';
    } else if (mode === 'backward') {
      if (currentStep === 0) return 'Ready for forward pass';
      if (currentStep === 1) return 'Forward pass done, ready for backpropagation';
      if (currentStep === 2) return 'Backpropagation done, ready to update weights';
      return 'Weight update completed';
    }
    return '';
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <Brain className="w-8 h-8 text-blue-600" />
            <h1 className="text-3xl font-bold text-gray-800">Neural Network Backpropagation Visualizer</h1>
          </div>
          <p className="text-gray-600">Interactive visualization to understand how neural networks learn through backpropagation</p>
        </div>

        {/* Controls */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Mode Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Mode</label>
              <select 
                value={mode} 
                onChange={(e) => setMode(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="forward">Forward Pass Only</option>
                <option value="backward">Full Backpropagation</option>
              </select>
            </div>

            {/* Activation Function */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Activation Function</label>
              <select 
                value={activationFunction} 
                onChange={(e) => setActivationFunction(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="sigmoid">Sigmoid</option>
                <option value="relu">ReLU</option>
                <option value="tanh">Tanh</option>
              </select>
            </div>

            {/* Learning Rate */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Learning Rate: {learningRate}
              </label>
              <input
                type="range"
                min="0.01"
                max="1"
                step="0.01"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>

          {/* Input Values */}
          <div className="mt-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">Training Data</label>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-500 mb-1">Input Values</label>
                <div className="flex gap-2">
                  {trainingData.input.map((val, i) => (
                    <input
                      key={i}
                      type="number"
                      step="0.1"
                      value={val}
                      onChange={(e) => {
                        const newInput = [...trainingData.input];
                        newInput[i] = parseFloat(e.target.value) || 0;
                        setTrainingData({...trainingData, input: newInput});
                      }}
                      className="w-20 px-2 py-1 border border-gray-300 rounded text-sm"
                    />
                  ))}
                </div>
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Target Output</label>
                <input
                  type="number"
                  step="0.1"
                  value={trainingData.target[0]}
                  onChange={(e) => {
                    setTrainingData({...trainingData, target: [parseFloat(e.target.value) || 0]});
                  }}
                  className="w-20 px-2 py-1 border border-gray-300 rounded text-sm"
                />
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 mt-6">
            <button
              onClick={stepForward}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Play className="w-4 h-4" />
              Step Forward
            </button>
            <button
              onClick={reset}
              className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </button>
            <button
              onClick={() => setShowHints(!showHints)}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            >
              <BookOpen className="w-4 h-4" />
              {showHints ? 'Hide' : 'Show'} Hints
            </button>
          </div>

          {/* Step Description */}
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <p className="text-blue-800 font-medium">Current Step: {getCurrentStepDescription()}</p>
          </div>
        </div>

        {/* Main Visualization */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <svg ref={svgRef} className="w-full border border-gray-200 rounded"></svg>
        </div>

        {/* Learning Hints */}
        {showHints && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4">Learning Hints</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-green-50 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">Forward Pass</h4>
                <p className="text-green-700 text-sm">
                  Data flows from input to output. Each neuron calculates: 
                  <br />
                  <code>activation = σ(Σ(weight × input) + bias)</code>
                </p>
              </div>
              <div className="p-4 bg-red-50 rounded-lg">
                <h4 className="font-semibold text-red-800 mb-2">Backpropagation</h4>
                <p className="text-red-700 text-sm">
                  Gradients flow backward using chain rule:
                  <br />
                  <code>∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w</code>
                </p>
              </div>
              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-2">Weight Updates</h4>
                <p className="text-blue-700 text-sm">
                  Weights are updated to minimize loss:
                  <br />
                  <code>new_weight = old_weight - learning_rate × gradient</code>
                </p>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <h4 className="font-semibold text-purple-800 mb-2">Visual Cues</h4>
                <p className="text-purple-700 text-sm">
                  • Thicker lines = larger weights/gradients
                  <br />
                  • Red glow = gradient flow during backprop
                  <br />
                  • Node color intensity = activation strength
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default NeuralNetworkVisualizer;