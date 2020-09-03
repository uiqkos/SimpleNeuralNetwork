using System;
using System.Collections.Generic;
using System.Linq;
using SimpleNNProject.Modules;

namespace SimpleNNProject {
    class Program {
        static void Main(string[] args) {
            
            NeuralNetwork nn = new NeuralNetwork(
                2, new []{3, 3}, 1, 
                ActivationFunctionType.Sigmoid,
                learningRate: 0.5, momentum: 0.6
            );

            List<Tuple<double[], double[]>> trainMatrix = new List<Tuple<double[], double[]>>() {
                Tuple.Create(new[] {0, 0d}, new[] {0d}), 
                Tuple.Create(new[] {0, 1d}, new[] {1d}), 
                Tuple.Create(new[] {1, 0d}, new[] {1d}), 
                Tuple.Create(new[] {1, 1d}, new[] {0d})
            };
            
            nn.Fit(trainMatrix, 100000);
            
            for (int i = 0; i < 4; i++) {
                nn.ForwardPropagate(trainMatrix[i].Item1);
                var predicted = nn.Get().ToList();
                var excepted = trainMatrix[i].Item2;
                Console.WriteLine("Test #" + (i + 1) );
                Console.WriteLine("Probability: " + predicted[0] + " excepted " + excepted[0]);
            }
        }
    }
}