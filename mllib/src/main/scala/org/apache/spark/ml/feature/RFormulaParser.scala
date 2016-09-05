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

package org.apache.spark.ml.feature

import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.mutable
import scala.util.parsing.combinator.RegexParsers

/**
  * Represents a parsed R formula.
  */
private[ml] case class ParsedRFormula(label: ColumnRef, terms: Seq[Term]) {
  /**
    * Resolves formula terms into column names. A schema is necessary for inferring the meaning
    * of the special '.' term. Duplicate terms will be removed during resolution.
    */
  def resolve(schema: StructType): ResolvedRFormula = {
    val dotTerms = expandDot(schema)
    var includedTerms = Seq[Seq[String]]()
    terms.foreach {
      case spline: Spline =>
        includedTerms :+= Seq(spline.col.value)
      case col: ColumnRef =>
        includedTerms :+= Seq(col.value)
      case ColumnInteraction(cols) =>
        includedTerms ++= expandInteraction(schema, cols)
      case Dot =>
        includedTerms ++= dotTerms.map(Seq(_))
      case Deletion(term: Term) =>
        term match {
          case inner: ColumnRef =>
            includedTerms = includedTerms.filter(_ != Seq(inner.value))
          case ColumnInteraction(cols) =>
            val fromInteraction = expandInteraction(schema, cols).map(_.toSet)
            includedTerms = includedTerms.filter(t => !fromInteraction.contains(t.toSet))
          case Dot =>
            // e.g. "- .", which removes all first-order terms
            includedTerms = includedTerms.filter {
              case Seq(t) => !dotTerms.contains(t)
              case _ => true
            }
          case _: Deletion =>
            throw new RuntimeException("Deletion terms cannot be nested")
          case _: Intercept =>
        }
      case _: Intercept =>
    }
    ResolvedRFormula(label.value, includedTerms.distinct, hasIntercept)
  }

  def explode_spline_terms_string(formula : String) : String ={
    val spline_terms = terms.collect{ case spline: Spline => spline }
    explode_spline_terms(formula, spline_terms)
  }

  def explode_spline_terms(formula : String, spline_terms : Seq[Spline]) : String = spline_terms.length match  {
    case 1 =>
      val spline : Spline = spline_terms(0)
      formula.replaceFirst("""\+.s\(.*?\)""",(0 to (stringListToVector(spline.knots.location).length-spline.basis.order)-1).foldLeft(""){ (s : String , number : Int) => s+"+ "+spline.col.value+"_smooth_"+number.toString+" "})
    case _ =>
      val spline : Spline = spline_terms(0)
      explode_spline_terms(formula.replaceFirst("""\+.s\(.*?\)""",(0 to (stringListToVector(spline.knots.location).length-spline.basis.order)-1).foldLeft(""){ (s : String , number : Int) => s+"+ "+spline.col.value+"_smooth_"+number.toString+" "}), spline_terms.tail)
  }

  def explode_spline_terms : String = {
    val spline_terms = terms.collect{ case spline: Spline => spline }
    val exploded_terms = spline_terms.map{spline: Spline => ColumnInteraction(Seq(spline.col))}
    val newTerms = terms.diff(spline_terms).union(spline_terms.flatMap{spline: Spline => explode_spline(spline)})
    deparse(newTerms)
  }

  private def deparse(new_terms : Seq[Term]) : String = {
    new_terms.tail.foldLeft(label.value+" ~ "+transform_term(new_terms(0), true)){(s : String, term : Term) => s+transform_term(term, false)}
  }

  private def transform_term(single_term : Term, first : Boolean) : String = {
    single_term match {
      case col: ColumnRef => if(first){col.value} else {" + "+col.value}
      case Dot => "."
      case Deletion(term: Term) => " - "+term
      case ColumnInteraction(col: Seq[ColumnRef]) => if(first){col(0).value} else {" + "+col(0).value}
    }
  }

  def explode_spline(spline : Spline) : List[Term] = {
    List.tabulate(stringListToVector(spline.knots.location).length-spline.basis.order){index => ColumnInteraction(Seq(ColumnRef(spline.col.value+"_smooth_"+index)))}
  }

  def transform_spline(dataset: Dataset[_]): Dataset[_] = {
    val spline_terms = terms.collect { case spline: Spline => spline }
    println("starting adding terms")
    if(spline_terms.isEmpty){dataset}else{add_spline(spline_terms, dataset)}


  }

  def add_spline(spline_terms: Seq[Spline], dataset: Dataset[_]): Dataset[_] = spline_terms.length match {
    case 1 =>
      val spline: Spline = spline_terms(0)
      val smooth_term_name = spline.col.value + "_smooth"
      val q = dataset.withColumn(smooth_term_name, toSplineFunctionVectorUdf(col(spline_terms(0).col.value),lit(spline.basis.order),lit(spline.knots.location)))
      println("order")
      println(spline.basis.order)
      println("knots length")
      println(println(stringListToVector(spline.knots.location).length))
      println("knots length-1")
      println(println(stringListToVector(spline.knots.location).length-1))
      println("knots length- basis order")
      println((stringListToVector(spline.knots.location).length-1)-spline.basis.order)
      add_individual_terms(smooth_term_name, 0, (stringListToVector(spline.knots.location).length-1)-spline.basis.order, q).drop(smooth_term_name)

    case _ =>
      val spline: Spline = spline_terms(0)
      val smooth_term_name = spline.col.value + "_smooth"
      println("die lange der knots_1")
      val q = dataset.withColumn(smooth_term_name, toSplineFunctionVectorUdf(col(spline_terms(0).col.value), lit(spline.basis.order),lit(spline.knots.location)))
      val qq = add_individual_terms(smooth_term_name, 0, (stringListToVector(spline.knots.location).length-1)-spline.basis.order, q)
      add_spline(spline_terms.tail, qq.drop(smooth_term_name))

  }

  def add_individual_terms(smooth_term_name : String, index : Int ,smooth_terms_n : Int, dataset: Dataset[_]): Dataset[_] ={
    if (index == smooth_terms_n) {
      dataset.withColumn(smooth_term_name+"_"+index,col(smooth_term_name)(index))
    } else{
      println("adding_smooth_terms_last")
      println(index+1)
      println(smooth_terms_n)
      add_individual_terms(smooth_term_name, index+1, smooth_terms_n, dataset.withColumn(smooth_term_name+"_"+index,col(smooth_term_name)(index)) )
    }
  }

  def stringListToVector(knotListString : String): Vector[Double] = {
    (knotListString.split(",").map(_.toDouble)).toVector
  }

  val toSplineFunctionVector: (Double, Int, String) => Vector[Double] = {
    (x, order, knotsList) => Vector.tabulate(stringListToVector(knotsList).length-order){ index => b(order,index,x,stringListToVector(knotsList)) }
  }

  val toSplineFunctionVectorUdf = udf(toSplineFunctionVector)

  //TODO: do not need to evaluate basis functions where they are zero
  def b(k: Int, i: Int, x: Double, knots: Vector[Double]): Double = k match {
    case 1 =>
      if (knots(i) <= x && x < knots(i + 1)) return 1 else 0
    case _ =>
      (x - knots(i)) / (knots(i + k - 1) - knots(i)) * b(k - 1, i, x, knots) +
        (knots(i + k) - x) / (knots(i + k) - knots(i + 1)) * b(k - 1, i + 1, x, knots)
  }

  /** Whether this formula specifies fitting with response variable. */
  def hasLabel: Boolean = label.value.nonEmpty

  /** Whether this formula specifies fitting with an intercept term. */
  def hasIntercept: Boolean = {
    var intercept = true
    terms.foreach {
      case Intercept(enabled) =>
        intercept = enabled
      case Deletion(Intercept(enabled)) =>
        intercept = !enabled
      case _ =>
    }
    intercept
  }

  // expands the Dot operators in interaction terms
  private def expandInteraction(
                                 schema: StructType, terms: Seq[InteractableTerm]): Seq[Seq[String]] = {
    if (terms.isEmpty) {
      return Seq(Nil)
    }

    val rest = expandInteraction(schema, terms.tail)
    val validInteractions = (terms.head match {
      case Dot =>
        expandDot(schema).flatMap { t =>
          rest.map { r =>
            Seq(t) ++ r
          }
        }
      case ColumnRef(value) =>
        rest.map(Seq(value) ++ _)
    }).map(_.distinct)

    // Deduplicates feature interactions, for example, a:b is the same as b:a.
    var seen = mutable.Set[Set[String]]()
    validInteractions.flatMap {
      case t if seen.contains(t.toSet) =>
        None
      case t =>
        seen += t.toSet
        Some(t)
    }.sortBy(_.length)
  }

  // the dot operator excludes complex column types
  private def expandDot(schema: StructType): Seq[String] = {
    schema.fields.filter(_.dataType match {
      case _: NumericType | StringType | BooleanType | _: VectorUDT => true
      case _ => false
    }).map(_.name).filter(_ != label.value)
  }
}

/**
  * Represents a fully evaluated and simplified R formula.
  *
  * @param label        the column name of the R formula label (response variable).
  * @param terms        the simplified terms of the R formula. Interactions terms are represented as Seqs
  *                     of column names; non-interaction terms as length 1 Seqs.
  * @param hasIntercept whether the formula specifies fitting with an intercept.
  */
private[ml] case class ResolvedRFormula(
                                         label: String, terms: Seq[Seq[String]], hasIntercept: Boolean) {

  override def toString: String = {
    val ts = terms.map {
      case t if t.length > 1 =>
        s"${t.mkString("{", ",", "}")}"
      case t =>
        t.mkString
    }
    val termStr = ts.mkString("[", ",", "]")
    s"ResolvedRFormula(label=$label, terms=$termStr, hasIntercept=$hasIntercept)"
  }
}

/**
  * R formula terms. See the R formula docs here for more information:
  * http://stat.ethz.ch/R-manual/R-patched/library/stats/html/formula.html
  */
private[ml] sealed trait Term

/** A term that may be part of an interaction, e.g. 'x' in 'x:y' */
private[ml] sealed trait InteractableTerm extends Term

/* R formula reference to all available columns, e.g. "." in a formula */
private[ml] case object Dot extends InteractableTerm

/* R formula reference to a column, e.g. "+ Species" in a formula */
private[ml] case class ColumnRef(value: String) extends InteractableTerm

/* R formula interaction of several columns, e.g. "Sepal_Length:Species" in a formula */
private[ml] case class ColumnInteraction(terms: Seq[InteractableTerm]) extends Term

/* R formula intercept toggle, e.g. "+ 0" in a formula */
private[ml] case class Intercept(enabled: Boolean) extends Term

/* R formula deletion of a variable, e.g. "- Species" in a formula */
private[ml] case class Deletion(term: Term) extends Term

/* R formula for GAM smooth term definition e.g. s(x) */
private[ml] case class Spline(col: ColumnRef, knots: KnotLocationList, basis: Basis) extends Term

/* R formula for knots location definiton for smooth terms */
private[ml] case class KnotLocationList(location : String)

/* R formula for knots provided for a smooth term */
private[ml] case class Knots(number: Int)

/* R formula for knots provided for a smooth term */
private[ml] case class Basis(order: Int)

/**
  * Limited implementation of R formula parsing. Currently supports: '~', '+', '-', '.', ':'.
  */
private[ml] object RFormulaParser extends RegexParsers {
  private val intercept: Parser[Intercept] =
    "([01])".r ^^ { case a => Intercept(a == "1") }

  private val spline: Parser[Spline] = "s(" ~ columnRef ~ "," ~ basis ~ "," ~ knotLocationList ~ ")" ^^ { case a ~ col ~ b ~ basis ~ c ~ knotLocationList ~ d => Spline(col, knotLocationList, basis) }

  private val columnRef: Parser[ColumnRef] =
    "([a-zA-Z]|\\.[a-zA-Z_])[a-zA-Z0-9._]*".r ^^ { case a => ColumnRef(a) }

  private val knots: Parser[Knots] = "k = " ~ "[0-9]*".r ^^ { case a ~ b => Knots(b.toInt) }

  private val knotLocationList : Parser[KnotLocationList] =
    "knots = c(" ~ "([0-9]*\\.[0-9]+)(\\,( )*([0-9]*\\.[0-9]+))*".r ~ ")" ^^ { case a ~ b ~ c => KnotLocationList(b) }



  //private val knots: Parser[KnotList] = "List(" ~ ")" ^^ {case a }

  private val basis: Parser[Basis] = "bs = " ~ ("cc" | "cs" | "ts") ^^ { case a ~ b => Basis(4) }

  private val empty: Parser[ColumnRef] = "" ^^ { case a => ColumnRef("") }

  private val label: Parser[ColumnRef] = columnRef | empty

  private val dot: Parser[InteractableTerm] = "\\.".r ^^ { case _ => Dot }

  private val interaction: Parser[List[InteractableTerm]] = rep1sep(columnRef | dot, ":")

  private val term: Parser[Term] = intercept | spline |
    interaction ^^ { case terms => ColumnInteraction(terms) } | dot | columnRef

  private val terms: Parser[List[Term]] = (term ~ rep("+" ~ term | "-" ~ term)) ^^ {
    case op ~ list => list.foldLeft(List(op)) {
      case (left, "+" ~ right) => left ++ Seq(right)
      case (left, "-" ~ right) => left ++ Seq(Deletion(right))
    }
  }

  private val formula: Parser[ParsedRFormula] =
    (label ~ "~" ~ terms) ^^ { case r ~ "~" ~ t => ParsedRFormula(r, t) }

  def parse(value: String): ParsedRFormula = parseAll(formula, value) match {
    case Success(result, _) => result
    case failure: NoSuccess => throw new IllegalArgumentException(
      "Could not parse formula: " + value)
  }
}
