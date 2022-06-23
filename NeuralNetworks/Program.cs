using System;
using NeuralNetworks.Utilities;

namespace NeuralNetworks
{
    class Program
    {
        static void Main(string[] args)
        {
            ArtificialNeuralNetwork ann = new(28 * 28, 10, 16, 16);
            MnistReader mnistReader = new();
            
            MnistReader.DigitImage image = mnistReader.PullDigit();
            
            // double[] result = ann.Perform(image.SpaghettifyAndNormalize());
            // Console.WriteLine("Expected output: " + image.label);
            // for (int i = 0; i < 10; i++)
            //     Console.WriteLine($"{i}: {result[i]:P}");
        }
    }
}