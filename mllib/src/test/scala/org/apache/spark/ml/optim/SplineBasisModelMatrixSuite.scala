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

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.{Instance, SmoothTerm}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD

class SplineBasisModelMatrixSuite extends SparkFunSuite with MLlibTestSparkContext {

  private var x: RDD[Double] = _
  private var y: RDD[Double] = _
  private var instances: RDD[Instance] = _
  private var instancesSmooth: RDD[SmoothTerm] = _
  private var instancesConstLabel: RDD[Instance] = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    /*
       R code:

       A <- matrix(c(0, 1, 2, 3, 5, 7, 11, 13), 4, 2)
       b <- c(17, 19, 23, 29)
       w <- c(1, 2, 3, 4)
     */

    val dataset= spark.createDataFrame(sc.parallelize(Seq(
      Instance(17.0, 1.0, Vectors.dense(0.0, 5.0).toSparse),
      Instance(19.0, 2.0, Vectors.dense(1.0, 7.0)),
      Instance(23.0, 3.0, Vectors.dense(2.0, 11.0)),
      Instance(29.0, 4.0, Vectors.dense(3.0, 13.0))
    ), 2))



  test("SPLINE TRANS TEST SUITE") {

    println("--------------before--------------")

    instances.foreach(p => println(p))

    val smooth_transformer_one = new SplineBasisModelMatrix(4, Vectors.dense(0.0,0.1,0.2,0.3,0.4,0.5,0.6,
      0.7,0.8,0.9,1.0)
    )




    val expected = dataset
    val actual = dataset
    //assert(actual ~== expected absTol 1e-4)
    assert(actual == expected)






    }
  }
}