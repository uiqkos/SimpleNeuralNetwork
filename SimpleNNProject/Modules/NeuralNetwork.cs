using System;
using System.Collections.Generic;
using System.Linq;
using static System.Math;

namespace SimpleNNProject.Modules {
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
            ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid,
            double learningRate = 0.25,
            double momentum = 0.9
        ) {
            InputLayer = new List<Neuron>();
            HiddenLayers = new List<List<Neuron>>();
            OutputLayer = new List<Neuron>();
            
            LearningRate = learningRate;
            Momentum = momentum;

            Func<double, double> activationFunction;
            Func<double, double> dActivationFunction;
            
            switch (activationFunctionType) {
                case ActivationFunctionType.Sigmoid:
                    activationFunction = x => 1f / (1 + Exp(-x));
                    dActivationFunction = x => x * (1 - x);
                    break;
                
                case ActivationFunctionType.HyperbolicTangent:
                    activationFunction = x => (Exp(x) - Exp(-x)) / Exp(x) + Exp(-x);
                    dActivationFunction = x => 1 - x*x;
                    break;
                
                default:
                    activationFunction = x => x;
                    dActivationFunction = x => 1;
                    break;
            }
            
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
        public void Fit(
            IEnumerable<Tuple<double[], double[]>> trainMatrix, 
            int epochCount
        ) {
            var trainMatrixEn = trainMatrix.ToList();
            for (int epoch = 0; epoch < epochCount; epoch++) {
                foreach (var (input, except) in trainMatrixEn) {
                    
                    ForwardPropagate(input);
                    
                    // Back propagate
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
        public void ForwardPropagate(IEnumerable<double> inputValues) {
            
            foreach (var (neuron, input) in InputLayer.Zip(inputValues)) 
                neuron.Value = input;

            foreach(var layer in HiddenLayers)
                layer.ForEach(neuron => neuron.CalculateValue());
            
            OutputLayer.ForEach(neuron => neuron.CalculateValue());
        }
        public IEnumerable<double> Get() {
            return OutputLayer.Select(neuron => neuron.Value); 
        }

        public IEnumerable<double> Predict(IEnumerable<double> input) {
            ForwardPropagate(input);
            return Get();
        }
        #endregion
        
    }

    public enum ActivationFunctionType {
        Sigmoid,
        HyperbolicTangent, // not working
        Linear
    }
}