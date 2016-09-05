package org.apache.spark.ml.feature


import org.apache.spark.sql._

//todo: run with irls
//todo: what if no splines supplied?
//todo: is the basis correct?
//todo: run with random market data

class RSmoothFormula () extends RFormula {

  def fit2(dataset: Dataset[_]): RSmoothFormula = {
    require(isDefined(formula), "Formula must be defined first.")
    val parsedFormula = RFormulaParser.parse($(formula))

    val data = parsedFormula.transform_spline(dataset)

    data.schema.printTreeString()
    data.show()

    new RSmoothFormula()
  }

  def smoothTransformer123(dataset: Dataset[_]) : (Dataset[_],String) = {
    val parsedFormula = RFormulaParser.parse($(formula))
    (parsedFormula.transform_spline(dataset).toDF(),parsedFormula.explode_spline_terms)
  }

  def formulaTransformer() : Unit = {

  }


}
