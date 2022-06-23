using System;
using System.IO;

namespace NeuralNetworks
{
    public class MnistReader
    {
        private const int IMG_WIDTH = 28;
        private const int IMG_HEIGHT = 28;
        
        public readonly int NumImages;
        public readonly int NumRows;
        public readonly int NumCols;
        public readonly int NumLabels;

        private readonly FileStream labelsFileStream;
        private readonly BinaryReader labelsBinaryReader;
        
        private readonly FileStream imagesFileStream;
        private readonly BinaryReader imagesBinaryReader;

        private bool closed = false;
        
        public MnistReader()
        {
            labelsFileStream = new FileStream(@"C:\Users\kylet\Projects\C#\NeuralNetworks\NeuralNetworks\MNIST Data Set\t10k-labels.idx1-ubyte", FileMode.Open); // test labels
            imagesFileStream = new FileStream(@"C:\Users\kylet\Projects\C#\NeuralNetworks\NeuralNetworks\MNIST Data Set\t10k-images.idx3-ubyte", FileMode.Open); // test images

            labelsBinaryReader = new BinaryReader(labelsFileStream);
            imagesBinaryReader = new BinaryReader(imagesFileStream);

            imagesBinaryReader.ReadInt32(); // discard
            NumImages = imagesBinaryReader.ReadInt32();
            NumRows = imagesBinaryReader.ReadInt32();
            NumCols = imagesBinaryReader.ReadInt32();

            labelsBinaryReader.ReadInt32();   //discard
            NumLabels = labelsBinaryReader.ReadInt32();
        }

        public DigitImage PullDigit()
        {
            if (closed)
                throw new Exception("FileStreams and BinaryReaders are already closed. This object is unreadable!");
            
            byte[][] pixels = new byte[IMG_WIDTH][];
            for (int i = 0; i < IMG_WIDTH; ++i)
            {
                byte[] pixelRow = new byte[IMG_HEIGHT];
                for (int j = 0; j < IMG_HEIGHT; ++j)
                {
                    byte b = imagesBinaryReader.ReadByte();
                    pixelRow[j] = b;
                }

                pixels[i] = pixelRow;
            }
                

            byte label = labelsBinaryReader.ReadByte();
            
            return new DigitImage(pixels, label);
        }

        public void Close()
        {
            if (!closed) return;
            closed = true;

            labelsFileStream.Close();
            labelsBinaryReader.Close();

            imagesFileStream.Close();
            imagesBinaryReader.Close();
        }
        
        public class DigitImage
        {
            public readonly byte[][] pixels;
            public readonly byte label;

            public DigitImage(byte[][] pixels, byte label)
            {
                this.pixels = pixels;
                this.label = label;
            }

            public double[] SpaghettifyAndNormalize()
            {
                double[] spaghetti = new double[IMG_WIDTH * IMG_HEIGHT];

                for (int i = 0; i < IMG_WIDTH; i++)
                {
                    for (int j = 0; j < IMG_HEIGHT; j++)
                    {
                        byte pixel = pixels[i][j];
                        int index = IMG_WIDTH * i + j;
                        
                        spaghetti[index] = pixel / 255d;
                    }
                }
            
                return spaghetti;
            }

            public override string ToString()
            {
                string s = $"===< Number {label} >==\n";
                for (int i = 0; i < IMG_WIDTH; ++i)
                {
                    for (int j = 0; j < IMG_HEIGHT; ++j)
                    {
                        byte pixel = pixels[i][j];
                        if (pixel == 0)
                            s += " "; // white
                        else if (pixel == 255)
                            s += "O"; // black
                        else
                            s += "."; // gray
                    }
                    s += "\n";
                }
                return s;
            }
        }
    }
}