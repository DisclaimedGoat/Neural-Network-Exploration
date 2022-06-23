using System;

namespace NeuralNetworks.Utilities
{
    public static class MathUtils
    {
        public static double Sigmoid(double x) => 1 / (1 + Math.Pow(Math.E, -x));

        public static double SigmoidDerivative(double x)
        {
            double sig = Sigmoid(x);
            return sig * (1 - sig);
        }
    }
}