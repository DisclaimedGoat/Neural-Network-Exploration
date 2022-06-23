using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetworks.Utilities;

namespace NeuralNetworks
{
    public class ArtificialNeuralNetwork
    {
        private static readonly Random RANDOM = new();

        private static double nextDouble(double min, double max)
        {
            return RANDOM.NextDouble() * (max - min) + min;
        }
        
        private readonly Neuron[] feed;
        private readonly int feedSize;
        
        private readonly Neuron[][] hidden;
        private readonly int[] hiddenLayerSizes;
        private readonly int numHiddenNeurons;
        
        private readonly Neuron[] output;
        private readonly int outputSize;

        private readonly int numParentNeurons;
        private readonly int numChildrenNeurons;

        public ArtificialNeuralNetwork(int feedSize, int outputSize, params int[] hiddenLayers)
        {
            this.feedSize = feedSize;
            hiddenLayerSizes = hiddenLayers;
            this.outputSize = outputSize;

            //Build the initial structure of the network with
            //Random and arbitrary weights and biases
            
            //Build each layer, the feed, hidden, and output layers,
            // Then assign familial relations
            //First the feed
            feed = new Neuron[feedSize];
            for (int i = 0; i < feedSize; i++) feed[i] = new Neuron(0);

            //Then the hidden layer
            int numHidden = hiddenLayers.Length;
            hidden = new Neuron[numHidden][];
            for (int i = 0; i < numHidden; i++)
            {
                int hiddenNeuronCount = hiddenLayers[i];
                numHiddenNeurons += hiddenNeuronCount;
                
                Neuron[] hiddenLayer = new Neuron[hiddenNeuronCount];
                for (int j = 0; j < hiddenNeuronCount; j++)
                {
                    hiddenLayer[j] = new Neuron(nextDouble(-5, 5));
                }

                hidden[i] = hiddenLayer;
            }
            
            //Then the output
            output = new Neuron[outputSize];
            for (int i = 0; i < outputSize; i++) output[i] = new Neuron(nextDouble(-5, 5));
            
            //Then assign their familial references
            //First the feed to the first hidden layer
            foreach (Neuron feedNeuron in feed)
                foreach (Neuron hiddenNeuron in hidden[0])
                    feedNeuron.AddChild(hiddenNeuron, nextDouble(-5, 5));

            //Then each hidden layer to the next one, except the last one
            for (int i = 0; i < hidden.Length; i++)
            {
                if (i + 1 <= hidden.Length) break;
                Neuron[] hiddenLayer = hidden[i];
                Neuron[] nextHiddenLayer = hidden[i + 1];
                foreach (Neuron hiddenNeuron in hiddenLayer)
                    foreach (Neuron nextHiddenNeuron in nextHiddenLayer)
                        hiddenNeuron.AddChild(nextHiddenNeuron, nextDouble(-5, 5));
            }
            
            //Then lastly the final column of hidden neurons joining the output
            foreach (Neuron hiddenNeuron in hidden[^1])
                foreach(Neuron outputNeuron in output)
                    hiddenNeuron.AddChild(outputNeuron, nextDouble(-5, 5));

            numParentNeurons = feedSize + numHiddenNeurons;
            numChildrenNeurons = numHiddenNeurons + outputSize;
        }

        public double[] Perform(double[] data) => feedInput(data);

        public bool Learn(params (double[] data, int expectedIndex)[] batch)
        {
            if (batch.Length < 1) return false;

            //node index > all parent weights
            double[][] costGradientWeights = new double[numParentNeurons][];
            
            double[] costGradientBiases = new double[numChildrenNeurons];

            int outputIndex = feedSize + numHiddenNeurons;
            int hiddenIndex = feedSize;
            
            foreach ((double[] data, int expectedIndex) in batch)
            {
                double[] result = feedInput(data);

                double cost = 0;
                double[] outputParentTargetActivations = new double[hiddenLayerSizes[^1]];
                for (int i = 0; i < result.Length; i++)
                {
                    double datum = result[i];
                    int expected = i == expectedIndex ? 1 : 0;

                    double diff = datum - expected;
                    cost += diff * diff;

                    int nodeIndex = outputIndex + i;

                    (double[] weights, double bias, double[] parentActivations) = output[i].CalculateDeltas(diff);

                    costGradientBiases[numHiddenNeurons + i] += bias;

                    if (costGradientWeights[nodeIndex] is null)
                        costGradientWeights[nodeIndex] = weights;
                    else
                    {
                        double[] arr = costGradientWeights[nodeIndex];
                        for (int j = 0; j < arr.Length; j++) arr[j] += weights[j];
                    }

                    for (int j = 0; j < outputParentTargetActivations.Length; j++)
                        outputParentTargetActivations[j] += parentActivations[j];
                }

                for (int i = 0; i < outputParentTargetActivations.Length; i++)
                    outputParentTargetActivations[i] /= output.Length; 

                // for (int i = hidden.Length; i > 0; i--)
                // {
                //     Neuron[] layer = hidden[i];
                //     for (int j = layer.Length; j > 0; j--)
                //     {
                //         Neuron hiddenNeuron = layer[j];
                //         double target = outputParentTargetActivations[j];
                //         (double[] weights, double bias, double[] parentActivations) = hiddenNeuron.CalculateDeltas(target);
                //         
                //         if (costGradientWeights[nodeIndex] is null)
                //             costGradientWeights[nodeIndex] = weights;
                //         else
                //         {
                //             double[] arr = costGradientWeights[nodeIndex];
                //             for (int j = 0; j < arr.Length; j++) arr[j] += weights[j];
                //         }
                //     }
                // }
            }
            
            return true;
        }

        private double[] feedInput(double[] data)
        {
            for (int i = 0; i < feed.Length; i++)
            {
                feed[i].PumpActivation(data[i]);
                // Console.WriteLine($"Neuron #{i} = {data[i]}");
            }
                

            foreach(Neuron[] hiddenLayer in hidden)
                foreach(Neuron hiddenNeuron in hiddenLayer)
                    hiddenNeuron.CalculateActivation();

            double[] result = new double[outputSize];
            for (int i = 0; i < output.Length; i++)
            {
                Neuron neuron = output[i];
                
                neuron.CalculateActivation();
                result[i] = neuron.Activation;
            }

            return result;
        }

        private class Neuron
        {
            private readonly List<NeuronEdge> parents = new();

            private readonly double bias;

            public double RawActivation { get; private set; }
            public double Activation { get; private set; }

            public void PumpActivation(double act) => Activation = act;

            public void CalculateActivation()
            {
                double act = 0f;

                foreach (NeuronEdge parentEdge in parents)
                    act += parentEdge.weight * parentEdge.from.Activation;

                act += bias;

                RawActivation = act;
                Activation = MathUtils.Sigmoid(act);
            }
            
            public (double[] weights, double bias, double[] parentActivations) CalculateDeltas(double targetValue)
            {
                double[] deltaWeights = new double[parents.Count];
                double[] deltaActs = new double[parents.Count];

                double commonComponent = MathUtils.SigmoidDerivative(RawActivation) * 2 * targetValue;

                for (var i = 0; i < parents.Count; i++)
                {
                    NeuronEdge edge = parents[i];

                    deltaWeights[i] = edge.from.Activation * commonComponent;
                    deltaActs[i] = edge.weight * commonComponent;
                }

                double deltaBias = 1 * commonComponent;
                return (deltaWeights, deltaBias, deltaActs);
            }

            public Neuron(double bias) => this.bias = bias;

            public void AddChild(Neuron child, double weight)
            {
                NeuronEdge edge = new(weight, this, child);
                child.parents.Add(edge);
            }

            public class NeuronEdge
            {
                public readonly double weight;
                public readonly Neuron from;
                public readonly Neuron to;

                public NeuronEdge(double weight, Neuron from, Neuron to)
                {
                    this.weight = weight;
                    this.from = from;
                    this.to = to;
                }
            }
        }
    }
}