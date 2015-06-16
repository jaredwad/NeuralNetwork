using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
   class Layer
   {
      private int itteration;
      private int numInputs;

      private List<Neuron> neurons;
      private List<double> inputs;
      private List<double> outputs;

      public Layer()
      {
         itteration = 0;
         neurons = new List<Neuron>();
      }

      public Layer(int pNumNeurons, ActivationFunction pFunction, int pNumInputs) : this()
      {
         createLayer( pNumNeurons, pFunction, pNumInputs );
      }

      public void createLayer(int pNumNeurons, ActivationFunction pFunction, int pNumInputs)
      {
         numInputs = pNumInputs;

         for( int i = 0; i < pNumNeurons; ++i ) {
            addNeuron( new Neuron( numInputs, pFunction ) );
         }
      }

      public void addNeuron( Neuron pNeuron ) 
      {
         //Not sure if this is a good idea, it will slow the entire network down...
//         if( itteration != 0 ) { throw new MethodAccessException( string.Format( "Cannont add a neuron after training has begun, current itteration = {0}", itteration ) ); }
         neurons.Add( pNeuron );
      }

      public List<double> run(List<double> pInputs)
      {
         inputs = pInputs;
         outputs = new List<double>();
         foreach(Neuron n in neurons) {
            outputs.Add( n.run( inputs ) );
         }

         return outputs;
      }

      public List<List<double>> getWeightMatrix()
      {
         List<List<double>> weightMatrix = new List<List<double>>();

         for( int i = 0; i < numInputs; ++i ) {
            List<double> weights = new List<double>(neurons.Count);
            foreach( Neuron n in neurons ) {
               weights.Add( n.getWeight( i ) );
            }
            weightMatrix.Add( weights );
         }

         return weightMatrix;
      }

      public List<double> getErrors()
      {
         List<double> errors = new List<double>( neurons.Count );

         foreach( Neuron n in neurons ) { errors.Add( n.LastError ); }

         return errors;
      }

      public void updateLayer(List<double> pExpectedValues)
      {
         for( int i = 0; i < neurons.Count; ++i ) {
            neurons[i].refactorWeightsOutput( pExpectedValues[i] );
         }
      }

      public void updateLayer(List<List<double>> pWeightMatrix, List<double> pErrors)
      {
         for( int i = 0; i < neurons.Count; ++i ) {
            neurons[i].refactorWeightsHidden( pWeightMatrix[i], pErrors );
         }
      }
   }
}
