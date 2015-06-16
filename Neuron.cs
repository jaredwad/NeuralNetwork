using JUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
   class Neuron
   {
      double bias;
      ActivationFunction function;

      List<double> weights;
      List<double> lastInputs;
     
      double lastOutput;
      double lastError;

      public double LearningRate { get; set; }
      public double LastOutput { get { return lastOutput; } }
      public double LastError  { get { return lastError;  } }

      # region constructors
      public Neuron( int pNumWeights, ActivationFunction pFunction, double pLearningRate = .3, double pBias = -1)
      {
         LearningRate = pLearningRate;
         bias = pBias;
         function = pFunction;
         initializeWeights( pNumWeights );
      }

      public Neuron( List<double> pWeights, ActivationFunction pFunction, double pBias = -1 )
      {
         bias = pBias;
         function = pFunction;
         weights = new List<double>( pWeights ); //Clone weights
      }
      #endregion constructors

      #region methods

      #region public

      public double run(List<double> pInputs)
      {
         checkInputs( pInputs );
         lastInputs = new List<double>( pInputs );

         lastInputs.Insert( 0, -1 );

         double sum = bias * weights[0]; //Should this equal bias?

         for( int i = 1; i < weights.Count; ++i ) {
            sum += weights[i] * lastInputs[i];
         }

         lastOutput = function.compute( sum );

         return lastOutput;
      }

      public double getWeight( int pIndex )
      {
         if( pIndex >= weights.Count ) {
            throw new ArgumentException( string.Format(
               "index {0} is larger than the count of the weights ({1})"
               , pIndex, weights.Count ) );
         }

         return weights[pIndex + 1];
      }

      public void refactorWeightsOutput(double pExpected)
      {
         lastError = lastOutput * ( 1 - lastOutput ) * ( lastOutput - pExpected );
         for( int i = 0; i < weights.Count; ++i ) {
            double newWeight = (weights[i] - (LearningRate * lastError * lastInputs[i]));
            weights[i] = newWeight;
         }
      }

      public void refactorWeightsHidden( List<double> pWeights, List<double> pErrors )
      {
         //Get the "error" for this node
         lastError = 0;
         for( int i = 0; i < pWeights.Count; ++i ) {
            lastError += pWeights[i] * pErrors[i];
         }
         lastError = lastError * lastOutput * ( 1 - lastOutput );

         //Update the weight
         for( int i = 0; i < weights.Count; ++i ) {
            weights[i] -= LearningRate * lastError * lastInputs[i];
         }
      }

      #endregion public

      #region private

      private void checkInputs(List<double> pInputs)
      {
         if( weights.Count - 1 != pInputs.Count )
            throw new ArgumentException( string.Format( "Number of Inputs ({0}) not equal to the number of weights ({1})", pInputs.Count, weights.Count ) );
      }

      private void initializeWeights(int pNumWeights)
      {
         weights = new List<double>();

         for( int i = 0; i < pNumWeights + 1; ++i ) {
            weights.Add( Rand.Double() ); //Get a random number close to 0
            if( Rand.Bool() ) //Decide whether or not to make it negative
               weights[i] = -weights[i];
         }
      }

      #endregion private
      #endregion methods
   }

   public abstract class ActivationFunction
   {
      public abstract double compute( double pInput );

//      public abstract ActivationFunction Clone();
   }

   public class stepFunction : ActivationFunction
   {
      private double threshold;
      public stepFunction( double pThreshold = 1.0 ) { threshold = pThreshold; }
      public override double compute( double pInput ) { return pInput >= threshold ? 1 : 0; }
//      public override ActivationFunction Clone() { return new stepFunction( threshold ); }
   }

   public class SigmoidFunction : ActivationFunction
   {
      public override double compute( double pInput )
      {
         return 1 / ( 1 + Math.Pow( Math.E, -pInput ) );
      }

//      public override ActivationFunction Clone() { return new SigmoidFunction(); }
   }
}
