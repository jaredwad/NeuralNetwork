using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
   class Network
   {
      private List<Layer> layers;
      private NetworkInterpreter interpreter;

      public Network( NetworkInterpreter pInterpreter )
      {
         layers = new List<Layer>();
         interpreter = pInterpreter;
      }

      public Network(NetworkInterpreter pInterpreter, int pNumInputs
         , ActivationFunction pFunction, List<int>pNeuronsPerLayer) : this(pInterpreter)
      {
         createNetwork( pNumInputs, pFunction, pNeuronsPerLayer );
      }

      public void createNetwork(int pNumInputs, ActivationFunction pFunction, List<int>pNeuronsPerLayer)
      {
         int numInputs = pNumInputs;
         int numNeurons;

         for( int i = 0; i < pNeuronsPerLayer.Count; ++i ) {
            numNeurons = pNeuronsPerLayer[i];
            layers.Add( new Layer( numNeurons, pFunction, numInputs ) );
            numInputs = numNeurons;
         }
      }

      public bool runAndUpdate(List<double> pInputs, string pExpectedOutput)
      {
         string output = classifyInstance( pInputs );

         updateNetwork( interpreter.getoutputFromClass( pExpectedOutput ) );

         return output == pExpectedOutput;
      }

      public List<double> runOne( List<double> pInputs )
      {
         foreach( var l in layers ) {
            pInputs = l.run( pInputs );
         }

         return pInputs;
      }

      public string classifyInstance(List<double> pInputs)
      {
         return interpreter.interpret( runOne( pInputs ) );
      }

      public void updateNetwork(List<double> pExpectedValues)
      {
         int index = layers.Count - 1;

         layers[index].updateLayer( pExpectedValues );

         for(index--; index > 0; index-- ) {
            layers[index].updateLayer( layers[index + 1].getWeightMatrix()
               , layers[index + 1].getErrors() );
         }

      }
   }

   public abstract class NetworkInterpreter
   {
      protected List<string> classes;
      public abstract string interpret(List<double> pOutputs);
      public abstract List<double> getoutputFromClass( string pClass );
   }

   public class DiabetesInterpreter : NetworkInterpreter
   {
      public DiabetesInterpreter()
      {
         classes = new List<string>();
         classes.Add( "tested_positive" );
         classes.Add( "tested_negative" );
      }

      public override string interpret( List<double> pOutputs )
      {
         return classes[pOutputs.IndexOf( pOutputs.Max() )];
      }

      public override List<double> getoutputFromClass( string pClass )
      {
         List<double> correctOutput = new List<double>();

         for( int i = 0; i < classes.Count; ++i ) {
            correctOutput.Add( pClass == classes[i] ? 1.0 : 0 );
         }

         return correctOutput;
      }
   }

   public class IrisInterpreter : NetworkInterpreter
   {

      public IrisInterpreter()
      {
         classes = new List<string>();
         classes.Add( "Iris-setosa" );
         classes.Add( "Iris-versicolor" );
         classes.Add( "Iris-virginica" );
      }

      public override string interpret( List<double> pOutputs )
      {
         return classes[pOutputs.IndexOf( pOutputs.Max() )];
      }

      public override List<double> getoutputFromClass( string pClass )
      {
         List<double> correctOutput = new List<double>();

         for( int i = 0; i < classes.Count; ++i ) {
            correctOutput.Add(pClass == classes[i] ? 1.0 : 0);
         }

         return correctOutput;
      }
   }
}
