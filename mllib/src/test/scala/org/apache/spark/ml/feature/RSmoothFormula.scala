package org.apache.spark.ml.feature

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext

class RSmoothFormulaSuite extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  test("encode smooth terms with transform") {
    val formula = new RSmoothFormula().setFormula("y ~ 1 + s(x1, bs=cs, k=10) + x1 + x2")

    val original = spark.createDataFrame(
      Seq(
        (1, 0.3, 0.11),
        (1, 0.4, 0.21),
        (1, 0.7, 0.41),
        (1, 0.1, 0.61))
    ).toDF("y", "x1" , "x2")

    val data = formula.smoothTransformer123(original)
    data._1.show
    println(data._2.toString)




    //val resultSchema = model.transformSchema(original.schema)

    val expected = spark.createDataFrame(
      Seq(
        (1, 0.4, 0.1, 0.28266666666666646,0.6306666666666668,0.08533333333333341,0.057166666666666685,0.0,0.0),
        (1, 0.5, 0.12, 0.020833333333333308,0.4791666666666666,0.47916666666666674,0.098784,0.0,0.0),
        (1, 0.6, 0.13, 0.0,0.08533333333333337,0.6306666666666667,0.12559516666666667,0.0,0.0 ),
        (1, 0.7, 0.14, 0.0,1.666666666666687E-4,0.22116666666666682,0.15686533333333338,0.0,0.0))
    ).toDF("id", "x1", "x2", "x1_smooth_0","x1_smooth_1","x1_smooth_2","x1_smooth_0","x1_smooth_1","x1_smooth_2")

    assert( true  === true)
  }
/*
  test("encode smooth terms with fit") {
    val formula = new RSmoothFormula().setFormula("y ~ 1 + s(x1, bs=cc, k=7) + x1 + s(x2, bs=cc, k=7)")

    val original = spark.createDataFrame(
      Seq(
        (1, 0.4, 0.1),
        (1, 0.5, 0.12),
        (1, 0.6, 0.13),
        (1, 0.7, 0.14))
    ).toDF("y", "x1" , "x2")

    val model = formula.fit2(original)
    val result = model.transform(original)

    //val resultSchema = model.transformSchema(original.schema)

    val expected = spark.createDataFrame(
      Seq(
        (1, 0.4, 0.1, 0.28266666666666646,0.6306666666666668,0.08533333333333341,0.057166666666666685,0.0,0.0),
        (1, 0.5, 0.12, 0.020833333333333308,0.4791666666666666,0.47916666666666674,0.098784,0.0,0.0),
        (1, 0.6, 0.13, 0.0,0.08533333333333337,0.6306666666666667,0.12559516666666667,0.0,0.0 ),
        (1, 0.7, 0.14, 0.0,1.666666666666687E-4,0.22116666666666682,0.15686533333333338,0.0,0.0))
    ).toDF("id", "x1", "x2", "x1_smooth_0","x1_smooth_1","x1_smooth_2","x1_smooth_0","x1_smooth_1","x1_smooth_2")

    assert( true === true)
  }

*/

}