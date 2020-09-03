using System;

namespace SimpleNNProject.Modules {
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
            Weight = 2 * new Random().NextDouble() - 1;
        }
        #endregion
    }
}