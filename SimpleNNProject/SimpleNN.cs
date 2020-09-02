using System;
using System.Collections.Generic;
using System.Linq;

namespace SimpleNNProject {
    public class NeuralNetwork {
        #region Properties
        private List<Neuron> InputLayer { get; }
        private List<List<Neuron>> HiddenLayers { get; }
        private List<Neuron> OutputLayer { get; }

        public double Momentum { get; set; }
        public double LearningRate { get; set; }
        #endregion

        #region Constructor
        public NeuralNetwork(
            int inputSize, int[] hiddenSizes, int outputSize, 
            Func<double, double> activationFunction, 
            Func<double, double> dActivationFunction,
            double learningRate = 0.25,
            double momentum = 0.9
        ) {
            InputLayer = new List<Neuron>();
            HiddenLayers = new List<List<Neuron>>();
            OutputLayer = new List<Neuron>();
            
            LearningRate = learningRate;
            Momentum = momentum;

            // Input
            for (int i = 0; i < inputSize; i++) 
                InputLayer.Add(new Neuron(activationFunction, dActivationFunction));
            
            // Hidden 
            var firstHidden = new List<Neuron>();
            for (int i = 0; i < hiddenSizes[0]; i++) 
                firstHidden.Add(new Neuron(InputLayer, activationFunction, dActivationFunction));
            
            HiddenLayers.Add(firstHidden);
            
            for (int i = 1; i < hiddenSizes.Length; i++) {
                var layer = new List<Neuron>();
            
                for (var j = 0; j < hiddenSizes[i]; j++)
                    layer.Add(new Neuron(
                        HiddenLayers[i - 1], activationFunction, dActivationFunction
                    ));
                HiddenLayers.Add(layer);
            }
            
            // Output
            for (int i = 0; i < outputSize; i++)
                OutputLayer.Add(new Neuron(HiddenLayers[^1], activationFunction, dActivationFunction));
            
        }
        #endregion

        #region Train
        public void BackPropagate(IEnumerable<Tuple<double[], double[]>> trainMatrix, int epochCount) {
            for (int epoch = 0; epoch < epochCount; epoch++) {
                foreach (var (input, except) in trainMatrix) {
                     ForwardPropagate(input);
                    
                    foreach (var (neuron, excepted) in OutputLayer.Zip(except)) 
                        neuron.Gradient = 
                            neuron.Value * (1 - neuron.Value) * (excepted - neuron.Value);
                    
                    foreach (var hiddenLayer in HiddenLayers.AsEnumerable()!.Reverse()) 
                        hiddenLayer.ForEach(neuron => neuron.CalculateGradient());
                    
                    foreach (var hiddenLayer in HiddenLayers.AsEnumerable()!.Reverse()) 
                        hiddenLayer.ForEach(neuron => neuron.UpdateWeights(LearningRate, Momentum));
                    
                    OutputLayer.ForEach(neuron => neuron.UpdateWeights(LearningRate, Momentum));
                    
                }
            }
        }
        #endregion

        #region Predict
        public void ForwardPropagate(double[] input) {
            var i = 0;
            InputLayer.ForEach(neuron => neuron.Value = input[i++]);
            
            foreach(var layer in HiddenLayers)
                layer.ForEach(neuron => neuron.CalculateValue());
            
            OutputLayer.ForEach(neuron => neuron.CalculateValue());
        }
        public IEnumerable<double> Get() {
            return OutputLayer.Select(neuron => neuron.Value); 
        }
        #endregion
        
    }

    public class Neuron {
        #region Properties
        public List<Synapse> InputSynapses { get; set; }
        public List<Synapse> OutputSynapses { get; set; }
        public double Value { get; set; }
        public double Bias { get; set; }
        public double BiasDelta { get; set; }
        public double Gradient { get; set; }
        
        private readonly Func<double, double> _act;
        private readonly Func<double, double> _dact;
        #endregion

        #region Constructors
        public Neuron(
            Func<double, double> activationFunction, 
            Func<double, double> dActivationFunction
        ) {
            _act = activationFunction;
            _dact = dActivationFunction;
            
            Bias = 2 * new Random().NextDouble() - 1;
            BiasDelta = 0;
            
            InputSynapses = new List<Synapse>();
            OutputSynapses = new List<Synapse>();
        }

        public Neuron(
            IEnumerable<Neuron> inputNeurons, 
            Func<double, double> activationFunction, 
            Func<double, double> dActivationFunction
        ): this(activationFunction, dActivationFunction) {
            
            foreach (var inputNeuron in inputNeurons) {
                InputSynapses.Add(new Synapse(inputNeuron, this));
                inputNeuron.OutputSynapses.Add(new Synapse(inputNeuron, this));
            }
        }
        #endregion

        #region Methods
        public double CalculateValue() {
            return Value = _act(
                InputSynapses.Select(
                    synapse => synapse.Weight * synapse.InputNeuron.Value
                ).Sum() + Bias
            );
        }

        public double CalculateGradient() {
            return Gradient = 
                OutputSynapses.Sum(a => a.OutputNeuron.Gradient * a.Weight) * _dact(Value);
        }

        public void UpdateWeights(double learnRate, double momentum) {
            var prevDelta = BiasDelta;
            BiasDelta = learnRate * Gradient; 
            Bias += learnRate * Gradient + prevDelta * momentum;
            
            foreach (var synapse in InputSynapses) {
                synapse.WeightDelta = learnRate * Gradient * synapse.InputNeuron.Value;
                synapse.Weight += synapse.WeightDelta;
            }
        }
        #endregion
    }

    public class Synapse {
        #region Properties
        public Neuron InputNeuron { get; }
        public Neuron OutputNeuron { get; }
        public double Weight { get; set; }
        public double WeightDelta { get; set; } = 0;
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