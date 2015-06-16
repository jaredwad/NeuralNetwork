using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
   class Program
   {
      static void Main( string[] args )
      {
         int[] temp = { 20, 40, 120, 2};//4 in first layer 3 in last layer
         double testPercent = .3;


         Network net = createNetwork( 8, temp.ToList() );

         List<Tuple<List<double>, string>> data = getData( @"C:\Users\Jared Wadsworth\Documents\Visual Studio 2013\Projects\NeuralNetwork\NeuralNetwork\diabetes-normalized.csv" );

         data = data.OrderBy( c => JUtils.Rand.Int( data.Count ) ).ToList();

         List<Tuple<List<double>, string>> training = new List<Tuple<List<double>, string>>();

         //Get training data
         for(int i = 0; i < data.Count; ++i) {
            if( JUtils.Rand.Double() <= testPercent ) {
               training.Add( data[i] );
               data.RemoveAt( i );
               i--;
            }
         }


         int numDataItems = data.Count;
         List<double> accuracy = new List<double>();

         int correctCount = 0;

         //Run 100 itterations through the data
         for( int i = 0; i < 3000; ++i ) {
            correctCount = 0;
            for( int j = 0; j < numDataItems; ++j ) {
               correctCount += net.runAndUpdate( data[j].Item1, data[j].Item2 ) ? 1 : 0;
            }
            accuracy.Add( (double)correctCount / (double)numDataItems );
            Console.WriteLine( "Itteration: {0}, accuracy: {1}", i, accuracy[i] );
         }

         correctCount = 0;

         for( int j = 0; j < training.Count; ++j ) {
            string classification = net.classifyInstance( training[j].Item1 );
            correctCount += classification == training[j].Item2 ? 1 : 0;
         }

         Console.WriteLine("test Accuracy: {0}",(double)correctCount / (double)training.Count);

         //for( int i = 0; i < accuracy.Count; ++i ) {
         //   Console.WriteLine( "Itteration: {0}, accuracy: {1}", i, accuracy[i] );
         //}
         Console.ReadKey();

      }

      static List<Tuple<List<double>, string>> getData(string pFileName)
      {
         List<Tuple<List<double>,string>> data = new List<Tuple<List<double>, string>>();

         List<string> lines = System.IO.File.ReadAllLines( pFileName ).ToList();

         //Remove Headers
         lines.RemoveAt( 0 );

         foreach(string line in lines) {
            string[] parts = line.Split( new char[] { ',' } );
            int i;
            int count = parts.Count() - 1;
            List<double> numbers = new List<double>();

            for( i = 0; i < count; i++ ) {
               numbers.Add( double.Parse( parts[i] ) );
            }

            data.Add( new Tuple<List<double>, string>( numbers, parts[i] ) );
         }

         return data;
      }

      static Network createNetwork(int pNumInputs, List<int> pNeuronsPerLayer)
      {
         NetworkInterpreter interpreter = new DiabetesInterpreter();
         Network net = new Network( interpreter );
         net.createNetwork( pNumInputs, new SigmoidFunction(), pNeuronsPerLayer );

         return net;
      }

      static Network createNetwork(int pNumInputs, int pNumLayers)
      {
         List<int> neuronsPerLayer = new List<int>( pNumLayers );

         for( int i = 0; i < pNumLayers; ++i ) {
            neuronsPerLayer.Add( JUtils.Rand.Int( 8 ) );
         }

         return createNetwork( pNumInputs, neuronsPerLayer );
      }
   }
}
