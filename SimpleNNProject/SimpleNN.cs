

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace SimpleNNProject {
    public class NeuralNetwork {
        #region Properties
        private List<List<Neuron>> _layers = new List<List<Neuron>>();
        private readonly int _layerCount = 0;
        private int[] _sizes;
        private Func<double, double> _activationFunction;
        private Func<double, double> _detActivationFunction;
        
        public readonly double LearningRate;
        #endregion

        #region Constructor
        public NeuralNetwork(double learningRate, Func<double, double> activationFunction, Func<double, double> detActivationFunction, params int[] sizes) {
            _activationFunction = activationFunction;
            _detActivationFunction = detActivationFunction;
            _sizes = sizes;
            _layerCount = sizes.Length;
            
            LearningRate = learningRate;
            
            // Input
            var inputLayer = new List<Neuron>();
            for (int i = 0; i < sizes[0]; i++) inputLayer.Add(new Neuron(activationFunction, detActivationFunction));
            _layers.Add(inputLayer);
            

            // Hidden and output
            
            for (int i = 1; i < sizes.Length; i++) {
                var layer = new List<Neuron>();

                for (var j = 0; j < sizes[i]; j++)
                    layer.Add(new Neuron(
                        _layers[i - 1], 
                        activationFunction,
                        detActivationFunction
                    ));
                _layers.Add(layer);
                
                foreach (var neuron in _layers[i - 1]) 
                    neuron.OutputSynapses = _layers[i]
                        .Select(neuron1 => new Synapse(neuron, neuron1))
                        .ToList();

                
            }
            
        }
        #endregion

        public List<double> ForwardPropagate(int[] input) {
            for (int i = 0; i < input.Length; i++) _layers[0][i].Value = input[i];

            for (var i = 1; i < _layerCount; i++)
                foreach (var neuron in _layers[i]) neuron.Calculate();

            return _layers[^1].Select(neuron => neuron.Value).ToList();
        }
        
        public void BackPropagate(List<Tuple<int[], int[]>> trainMatrix, int epochCount) {
            for (int epoch = 0; epoch < epochCount; epoch++) {
                foreach (var (input, except) in trainMatrix) {
                    var y = ForwardPropagate(input);

                    // _layers[^1]
                    //     .Zip(except)
                    //     .ToList()
                    //     .ForEach(tuple => tuple.First.Gradient = tuple.Second - tuple.First.Value);
                    
                    var error = y.
                        Zip(except)
                        .Select(tuple => Math.Pow(tuple.First - tuple.Second, 2))
                        .Sum() / except.Length;

                    foreach (var neuron in _layers[^1]) {
                        neuron.Gradient = error * _detActivationFunction(neuron.Value);
                    }

                    foreach (var layer in _layers.SkipLast(1).Reverse())
                        foreach (var neuron in layer) {
                            neuron.UpdateGradient();
                            neuron.UpdateWeights(LearningRate, error);
                        }
                }
            }
        }

        public List<double> Weights() {
            var weights = new List<List<double>>();
            foreach (var layer in _layers) {
                weights.Add(layer
                    .Select(neuron => neuron.OutputSynapses.Select(synapse => synapse.Weight))
                    .SelectMany(w => w).ToList()
                );
            }

            return weights.SelectMany(w => w).ToList();
        }
        
        private List<double> SumList(List<double> first, List<double> second) => 
            first.Zip(second).Select(tuple => tuple.First + tuple.Second).ToList();

    }

    public class Neuron {
        #region Properties
        public List<Synapse> InputSynapses { get; set; }
        public List<Synapse> OutputSynapses { get; set; }
        public double Value { get; set; }
        public double Bias { get; set; } = 1;
        public double Gradient { get; set; }
        
        private Func<double, double> _activationFunction;
        private Func<double, double> _detActivationFunction;
        #endregion

        #region Constructors
        public Neuron(Func<double, double> activationFunction, Func<double, double> detActivationFunction) {
            _activationFunction = activationFunction;
            _detActivationFunction = detActivationFunction;
            Value = 0;
            InputSynapses = new List<Synapse>();
            OutputSynapses = new List<Synapse>();
        }

        public Neuron(List<Neuron> inputNeurons, Func<double, double> activationFunction, Func<double, double> detActivationFunction) {
            _activationFunction = activationFunction;
            _detActivationFunction = detActivationFunction;
            Value = 0;
            OutputSynapses = new List<Synapse>();
            InputSynapses = inputNeurons
                .Select((neuron, i) => new Synapse(
                    inputNeurons[i], neuron)
                ).ToList();
        }
        #endregion

        #region Methods
        public double Calculate() {
            return Value = _activationFunction(
                InputSynapses.Select(
                    synapse => synapse.Weight * synapse.InputNeuron.Value
                ).Sum() + Bias
            );
        }

        public void UpdateGradient() {
            Gradient = OutputSynapses
                .Select(synapse => synapse.Weight * synapse.OutputNeuron.Gradient)
                .Sum() * _detActivationFunction(Value);
        }

        public void UpdateWeights(double learningRate, double error) {
            foreach (var synapse in OutputSynapses) {
                var deltaWeight = error * 
                    (synapse.OutputNeuron.Gradient * synapse.InputNeuron.Value) // grad
                    + learningRate * synapse.DeltaWeight;
                // Console.WriteLine(synapse.OutputNeuron.Gradient);
                
                synapse.DeltaWeight = deltaWeight;
                synapse.Weight += deltaWeight;
            }
        }
        #endregion
    }

    public class Synapse {
        #region Properties
        public Neuron InputNeuron { get; set; }
        public Neuron OutputNeuron { get; set; }
        public double Weight { get; set; }
        public double DeltaWeight { get; set; } = 0;
        #endregion
        
        #region Constructor
        public Synapse(Neuron inputNeuron, Neuron outputNeuron) {
            InputNeuron = inputNeuron;
            OutputNeuron = outputNeuron;
            Weight = 2 * (new Random().NextDouble()) - 1;
        }
        #endregion
    }
    
}