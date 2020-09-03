using System;
using System.Collections.Generic;
using System.Linq;

namespace SimpleNNProject.Modules {
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
    
}