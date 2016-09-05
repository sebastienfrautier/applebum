/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.optim

import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD



private[ml] class SplineBasisModelMatrix(
                                        val order: Int,
                                        val knots : Vector
                                          ) extends Logging with Serializable {

/*
  val smoothUdf = udf((x, order, knots) => SmoothTerm(form_function_vector()))

  def smoothTransform(df: DataFrame, predictor_name : String) : DataFrame = {

    df.withColumn("x_smooth_terms", smoothUdf("x", order, knots))

  }
*/
  def smooth_transformer(instances : RDD[Instance], feature_index : Int ): RDD[Instance]= {

    instances.map(instance => Instance(instance.label, instance.weight, form_function_vector(feature_index,order, instance.features, knots)))


  }

  def form_function_vector(feature_index : Int, order : Int, x_values : Vector, knots : Vector): Vector = {

    Vectors.dense(Array.tabulate(knots.size-order)(b(order,_,x_values.apply(feature_index),knots)))

  }



  def b(k : Int, i : Int, x : Double, knots : Vector): Double = k match{
    case 1 =>
      if (knots(i) <= x && x < knots(i+1)) return 1 else 0
    case _ =>
      (x - knots(i)) / (knots(i + k - 1) - knots(i)) * b(k - 1, i, x, knots) +
        (knots(i + k) - x) / (knots(i + k) - knots(i + 1)) * b(k - 1, i + 1, x, knots)
  }

/*
  //BD class "cyclic.smooth" objects include matrix BD which transforms function values at
  // the knots to second derivatives at the knots.
  def getBD(knots : Vector): List[DenseMatrix[Double]] = {
    val nn  = knots.size
    val h : DenseVector[Double] = knots(1 to nn-1)-knots(0 to nn-2) // h<-x[2:n]-x[1:(n-1)]
    val n = nn-1
    val D = DenseMatrix.zeros[Double](n,n)
    val B = DenseMatrix.zeros[Double](n,n)
    B(0 to 0,0 to 0)      := ((h(n-1) + h(0)) /3)      // B[1,1]<-(h[n]+h[1])/3
    B(0 to 0,1 to 1)      := h(0)/6                    // B[1,2] <- h[1]/6
    B(0 to 0, n-1 to n-1) := h(n-1)/6             // B[1,n] <- h[n]/6

    D(0 to 0, 0 to 0)     := -((1/h(0))+(1/h(n-1)))       // D[1,1]<- -(1/h[1]+1/h[n])
    D(0 to 0, 1 to 1)     := 1/h(0)                   // D[1,2]<-1/h[1]
    D(0 to 0, n-1 to n-1) := 1/h(n-1)             // D[1,n]<-1/h[n]

    (2 to n-1).foreach{
      i =>
        B(i-1 to i-1, i-2 to i-2) := h(i-2)/6                 //B[i,i-1]<-h[i-1]/6
        B(i-1 to i-1, i-1 to i-1) := ((h(i-2)+h(i-1)) /3)        //B[i,i]<-(h[i-1]+h[i])/3
        B(i-1 to i-1, i to i)     := h(i-1)/6                     //B[i,i+1]<-h[i]/6

        D(i-1 to i-1, i-2 to i-2) := 1/h(i-2)                 //D[i,i-1]<-1/h[i-1]
        D(i-1 to i-1, i-1 to i-1) := -(1/h(i-2) + 1/h(i-1))   //D[i,i]<- -(1/h[i-1]+1/h[i])[
        D(i-1 to i-1, i to i)     := 1/h(i-1)                     //D[i,i+1]<- 1/h[i]

    }

    B(n-1 to n-1, n-2 to n-2) := h(n-2)/6               // B[n,n-1]<-h[n-1]/6
    B(n-1 to n-1, n-1 to n-1) := (h(n-2)+h(n-1))/3      // B[n,n]<-(h[n-1]+h[n])/3
    B(n-1 to n-1, 0 to 0)     := h(n-1)/6               // B[n,1]<-h[n]/6

    D(n-1 to n-1, n-2 to n-2) := 1/h(n-2)               // D[n,n-1]<-1/h[n-1]
    D(n-1 to n-1, n-1 to n-1) := -(1/h(n-2)+1/h(n-1))   // D[n,n]<- -(1/h[n-1]+1/h[n])
    D(n-1 to n-1, 0 to 0)     :=  1/h(n-1)               // D[n,1]<-1/h[n]

    List[DenseMatrix[Double]](B,D)

  }

*/
}



private[ml] class BasisMatrix(
                               val modelMatrix: DenseMatrix ,
                               val knots : Vector,
                               val cubic : Boolean) extends Serializable {

}





/*
private[ml] object SplineBasisModelMatrix {
}




private[ml] class SplineBasisModelMatrixModel(
                                             val cubic: DenseVector,
                                             val diagInvAtWA: DenseVector) extends Serializable {
}
*/