using System;
using System.Collections.Generic;
using System.Linq;

namespace SimpleNNProject {
    class Program {
        static void Main(string[] args) {
            Func<double, double> sigmoid = x => 1f / (1 + Math.Exp(-x));
            Func<double, double> detSigmoid = x => x * (1 - x);
            
            NeuralNetwork nn = new NeuralNetwork(
                2, new []{3, 3}, 1, 
                sigmoid, detSigmoid,
                0.5, 0.6
            );

            var trainMatrix = new List<Tuple<double[], double[]>> ();
            
            trainMatrix.Add(Tuple.Create(new[] {0, 0d}, new []{0d}));
            trainMatrix.Add(Tuple.Create(new[] {0, 1d}, new []{1d}));
            trainMatrix.Add(Tuple.Create(new[] {1, 0d}, new []{1d}));
            trainMatrix.Add(Tuple.Create(new[] {1, 1d}, new []{0d}));

            nn.BackPropagate(trainMatrix, 100000);
            
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