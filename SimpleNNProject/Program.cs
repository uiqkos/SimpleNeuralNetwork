using System;
using System.Collections.Generic;

namespace SimpleNNProject {
    class Program {
        static void Main(string[] args) {
            Func<double, double> sigmoid = x => (double) (1f / (1 + Math.Exp(-x)));
            Func<double, double> detSigmoid = x => x*(1 - x); //sigmoid(x) * (1 - sigmoid(x));    
            NeuralNetwork nn = new NeuralNetwork(1, sigmoid, detSigmoid, 2, 4, 1);
            
            // nn.Weights().ForEach(weight => Console.WriteLine(weight));
            
            var trainMatrix = new List<Tuple<int[], int[]>> ();
            
            trainMatrix.Add(Tuple.Create(new[] {0, 0}, new []{0}));
            trainMatrix.Add(Tuple.Create(new[] {0, 1}, new []{1}));
            trainMatrix.Add(Tuple.Create(new[] {1, 0}, new []{1}));
            trainMatrix.Add(Tuple.Create(new[] {1, 1}, new []{0}));

            nn.BackPropagate(trainMatrix, 10000);
            
            // nn.Weights().ForEach(weight => Console.WriteLine(weight));
            
            //
            // // Func<double, double> sigmoid = x => x;
            // // Func<double, double> detSigmoid = x => 1;
            //
            // NeuralNetwork neuralNetwork = new NeuralNetwork(0.5f, sigmoid, detSigmoid, 2, 3, 1);
            //
            // var trainMatrix = new List<Tuple<int[], int[]>> ();
            //
            // trainMatrix.Add(Tuple.Create(new[] {0, 0}, new []{0}));
            // trainMatrix.Add(Tuple.Create(new[] {0, 1}, new []{1}));
            // trainMatrix.Add(Tuple.Create(new[] {1, 0}, new []{1}));
            // trainMatrix.Add(Tuple.Create(new[] {1, 1}, new []{0}));
            //
            // neuralNetwork.BackPropagate(trainMatrix, 10);
            //
            // var testMatrix = trainMatrix;
            //
            //
            for (int i = 0; i < 4; i++) {
                var predicted = nn.ForwardPropagate(trainMatrix[i].Item1);
                var excepted = trainMatrix[i].Item2;
                Console.WriteLine("Test #" + (i + 1) );
                Console.WriteLine("Probability: " + predicted[0] + " excepted " + excepted[0]);
            }
        }
    }
}