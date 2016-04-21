import breeze.linalg._
import breeze.numerics._
import java.io._


object HelloWorld {
  def main(args: Array[String]): Unit = {

    //val y_vector = DenseVector[Double]((0.0 to 1.0 by .05).toArray)
    //val x_vector = DenseVector[Double]((0.0 to 1.0 by .05).toArray)


    val knots = DenseVector[Double]((.0 to 1.0 by .2).toArray)
    val xp = DenseVector[Double]((.0 to 1.0 by .01).toArray)
    val size = DenseVector[Double](1.42,1.58,1.78,1.99,1.99,1.99,2.13,2.13,2.13, 2.32,2.32,2.32,2.32,2.32,2.43,2.43,2.78,2.98,2.98)
    val wear = DenseVector[Double](4.0,4.2,2.5,2.6,2.8,2.4,3.2,2.4,2.6,4.8,2.9, 3.8,3.0,2.7,3.1,3.3,3.0,2.8,1.7)
    val pp = size-min(size)
    val x = pp/(max(pp))
    val lambda = 1e-8


    val (x_, y_, beta) = gam(wear, List(x,x),List(knots,knots))



    /*println("bd_list(0)")
    //println(bd_list(0))
    //println("bd_list(1)")
    //println(bd_list(1))


    //val X = predict_matrix_cyclic_smooth(x, knots, BD)

    //

    //val beta = Xa \ y
    // println(beta)
    val bd_list = getBD(knots)
    val BD = bd_list(0) \ bd_list(1)
    val X = predict_matrix_cyclic_smooth(x ,knots, BD)
    val Xa = DenseMatrix.vertcat(X, (matrix_square_root(bd_list(1).t * BD) * sqrt(lambda)))
    val y = DenseVector.vertcat(wear, DenseVector.zeros[Double](Xa.rows-wear.length))
    val model_coeff = Xa \ y
    val Xpredict = predict_matrix_cyclic_smooth(xp, knots, BD)
    //println(model_coeff)

    val qqq = Xpredict * (X \ wear)
    val sss = Xpredict * model_coeff

    //print(prs_fit(wear,x, knots, lambda))


    breeze.linalg.csvwrite(new File("/Users/kbrusch/dev/gam-fitting-in-spark/data/X.csv")
      ,qqq.asDenseMatrix )

    breeze.linalg.csvwrite(new File("/Users/kbrusch/dev/gam-fitting-in-spark/data/S.csv")
      ,sss.asDenseMatrix )

    println(try_lambda(wear, x, knots, lambda, 1 ,wear.length-1, List()))

    */
  }


  def try_lambda(y : DenseVector[Double], x : DenseVector[Double], knots : DenseVector[Double], lambda : Double, i : Int, n : Int, scores : List[Any]) : List[Any] = i match {

    case 60 =>
      val (xa, coeffs,ya) = prs_fit(y,x,knots,lambda)
      val tra = trA(xa, n)
      val rss = form_residuals_squared(xa, coeffs, ya, n)
      val gcv = n*rss/pow((n-tra),2)
      scores :+ (lambda,gcv)
    case _ =>
      val (xa, coeffs,ya) = prs_fit(y,x,knots,lambda)
      val tra = trA(xa, n)
      val rss = form_residuals_squared(xa, coeffs, ya, n)
      val gcv = n*rss/pow((n-tra),2)
      try_lambda(y, x, knots, lambda*1.5, i+1,n, scores :+ (lambda,gcv))

  }




    // TODO: just pass in BD
  def form_X(x : DenseVector[Double], knots : DenseVector[Double]) : DenseMatrix[Double] = {
    val bd_list = getBD(knots)
    val BD = bd_list(0) \ bd_list(1)
    predict_matrix_cyclic_smooth(x ,knots, BD)
  }

  def form_S(knots : DenseVector[Double]) :  DenseMatrix[Double] = {
    val bd_list = getBD(knots)
    val BD = bd_list(0) \ bd_list(1)
    bd_list(1).t * BD
  }


  def gam(response : DenseVector[Double], predictors : List[DenseVector[Double]], knots : List[DenseVector[Double]] ) = {
    val lambdas = DenseVector(0.0001, 0.0002)
    val (xa_, s_) = gam_setup(predictors, knots)
    fit_gam(response, xa_, s_,lambdas)
  }

  def gam_setup(predictors : List[DenseVector[Double]], knots : List[DenseVector[Double]]) :  (DenseMatrix[Double], List[DenseMatrix[Double]]) ={

    val Xa = predictors.zip(knots).map{ case (x_vector, knot_vector) => form_X(x_vector, knot_vector)}.foldLeft(DenseMatrix.ones[Double](19,1)){(agg,ele) => DenseMatrix.horzcat(agg,ele)}
    val S = knots.zipWithIndex.map{ case (knot_vector, index) => align_S(form_S(knot_vector),index, knots.length)}
    (Xa, S)
  }

  def align_S(S : DenseMatrix[Double], index : Int, max : Int) : DenseMatrix[Double] = {
      val full_S = DenseMatrix.zeros[Double](S.rows*max, S.cols)
      full_S(index*S.rows to ((index+1)*S.rows)-1, 0 to S.cols-1) := S
      full_S
  }

  def fit_gam(y: DenseVector[Double], Xa : DenseMatrix[Double], S_list : List[DenseMatrix[Double]], lambdas : DenseVector[Double]) =  {

    val S = S_list.zipWithIndex.map{ case (s_matrix, lambda_index) => s_matrix * lambdas(lambda_index)}.foldLeft(DenseMatrix.ones[Double](10,1)){(agg,ele) => DenseMatrix.horzcat(agg,ele)}
    println(S(::, 1 to S.cols-1))
    println(Xa(::, 1 to Xa.cols-1))
    println(matrix_square_root(S(::, 1 to S.cols-1)))
    val X1 = DenseMatrix.vertcat(Xa(::, 1 to Xa.cols-1), matrix_square_root(S(::, 1 to S.cols-1)))

    fitting_loop(X1, y, Xa.cols ,Xa.rows, DenseVector.vertcat(DenseVector(1.0), DenseVector.zeros[Double](X1.cols-1)) , false)

  }

  def fitting_loop(X1 : DenseMatrix[Double], y : DenseVector[Double], q : Int ,n : Int , beta : DenseVector[Double], norm : Boolean): (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = norm match {
    case true =>
      (X1, y, beta)
    case _ =>
      println("X1.rows")
      println(X1)
      println("beta")
      println(beta)
      val eta_ = (X1 * beta)
      println("eta_")
      println(eta_)
      val eta = eta_(0 to n-1)
      println("eta")
      println(eta)
      val mu = exp(eta)
      //n-y.length
      val z = DenseVector.vertcat((y-mu)/mu + eta, DenseVector.zeros[Double](X1.rows-y.length))
      println("z")
      println(z)
      println(X1)
      val new_beta = X1 \ z
      val tra_ = trA(X1, y.length-1)
      val norm = form_residuals_squared(X1, new_beta, z, y.length-1)
      fitting_loop(X1, y, q, n ,new_beta,false)
  }


  def prs_fit(y : DenseVector[Double], x : DenseVector[Double], knots : DenseVector[Double], lambda : Double) : (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {
    val bd_list = getBD(knots)
    val BD = bd_list(0) \ bd_list(1)
    val X = predict_matrix_cyclic_smooth(x ,knots, BD)
    val Xa = DenseMatrix.vertcat(X, (matrix_square_root(bd_list(1).t * BD) * sqrt(lambda)))
    val ya = DenseVector.vertcat(y, DenseVector.zeros[Double](Xa.rows-y.length))
    (Xa, Xa \ ya, ya)

  }



  def form_residuals_squared(X :DenseMatrix[Double], beta : DenseVector[Double], y : DenseVector[Double], n : Int): Double ={
    val vals = pow((y-X*beta),2)
    sum(vals(0 to n))
  }
  def trA(X : DenseMatrix[Double], n : Int) : Double = {
    val hat = diag(hat_matrix(X))
    val first_n_elements = hat(0 to n)
    sum(first_n_elements)
  }

  def hat_matrix(X : DenseMatrix[Double]) : DenseMatrix[Double] = {
    X*inv(X.t*X)*X.t
  }

  def hat_matrixQR(X : DenseMatrix[Double]) : DenseMatrix[Double] = {
    //val qr_matrix_object = qr(X)
    //val QR = qr_matrix_object.q * qr_matrix_object.r

    val n = X.rows
    val qr_matrix_object = qr(X)
    X



  }

  def splX(x_vector : DenseVector[Double], knots : DenseVector[Double]): DenseMatrix[Double]= {
    //val X = DenseMatrix.ones[Double](x_vector.length,2)
    //X(::,1) := x_vector
    //DenseMatrix.horzcat(X,cyclic_cubic_base(x_vector, knots, 4))
    cyclic_cubic_base(x_vector, knots, 4)

  }
  def splS(knots : DenseVector[Double]): DenseMatrix[Double]= {
    //val S = DenseMatrix.zeros[Double](knots.length,2)
    //DenseMatrix.horzcat(S,penalty_matrix(knots))
    penalty_matrix(knots)

  }


  def diffMatrix(matrix : DenseMatrix[Double], difference : Int ) : DenseMatrix[Double] = {
    matrix(::,*).map(column => diff(column, difference))
  }


  def diff(column_vector : DenseVector[Double], difference : Int ): DenseVector[Double] = difference match {
    case 1 => column_vector((0+1) to column_vector.length-1) - column_vector(0 to column_vector.length-1-1)
    case _ => diff(column_vector((0+1) to column_vector.length-1) - column_vector(0 to column_vector.length-1-1), difference-1)
  }


  def penalty_matrix(knots : DenseVector[Double]): DenseMatrix[Double] ={

    val diagonal = diag(DenseVector.ones[Double](knots.length))
    val diffM = diffMatrix(diagonal, 1)
    diffM.t * diffM

  }

  def matrix_square_root( matrix : DenseMatrix[Double]) : DenseMatrix[Double] ={
    val d = eig(matrix)
    d.eigenvectors*diag(sqrt(d.eigenvalues))*d.eigenvectors.t
  }

  def design_spline(x : DenseVector[Double] ,knots : DenseVector[Double], basis : Int): DenseMatrix[Double] = {
    generate_basis_matrix(x, knots, basis)
  }

  //https://en.wikipedia.org/wiki/B-spline
  //TODO: do not need to evaluate basis functions where they are zero
  def b(k : Int, i : Int, x : Double, knots : DenseVector[Double]): Double = k match{
    case 1 =>
      if (knots(i) <= x && x < knots(i+1)) return 1 else 0
    case _ =>
      (x - knots(i)) / (knots(i + k - 1) - knots(i)) * b(k - 1, i, x, knots) +
        (knots(i + k) - x) / (knots(i + k) - knots(i + 1)) * b(k - 1, i + 1, x, knots)
  }

  def cyclic_cubic_base(x_vector : DenseVector[Double], knots: DenseVector[Double], order : Int): DenseMatrix[Double] ={
    // knots are sorted
    val nk = knots.length-1
    val k1 = knots(0)


    // should be DenseVector.vertcat( k1 - (knots(nk) - knots((nk - order + 1) to (nk - 1))) , knots)
    // but IDE is weird af
    val wtf = knots((nk - order + 1) to (nk - 1)).map( ele => k1 - (knots(nk) - ele))
    val adjusted_knots = DenseVector.vertcat( wtf , knots)

    val xc = knots(nk - order + 1)
    val X1 = generate_basis_matrix(x_vector, adjusted_knots, order)
    val ind = x_vector :> xc


    x_vector(ind.argmax to ind.length-1) := x_vector(ind.argmax to ind.length-1).map(e => e-adjusted_knots.max - k1)

    val X2 = generate_basis_matrix(x_vector(ind.argmax to ind.length-1), adjusted_knots, order)

    X1(ind.argmax to ind.length-1, ::) := X1(ind.argmax to ind.length-1, ::) + X2
    X1
  }

  def b_deriv(k : Int, i : Int, x : Double , knots : DenseVector[Double]): Double = {
    (k-1)*((-(b(k-1,i+1,x,knots))/(knots(i+k)-knots(i+1))+((b(k-1,i,x,knots)/knots(i+k+1)-knots(i)))))
  }

  def generate_basis_matrix(x : DenseVector[Double], knots : DenseVector[Double], k : Int): DenseMatrix[Double]={

    val X = DenseMatrix.tabulate(x.length, knots.length - k){ case (i, j) =>  b(k, j, x(i), knots) }
    X

  }

  //BD class "cyclic.smooth" objects include matrix BD which transforms function values at
  // the knots to second derivatives at the knots.
  def getBD(knots : DenseVector[Double]): List[DenseMatrix[Double]] = {
    val nn  = knots.length
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




  def predict_matrix_cyclic_smooth(x : DenseVector[Double], knots : DenseVector[Double], BD : DenseMatrix[Double]) : DenseMatrix[Double] = {
    val j = x.copy
    val n = knots.length
    val h : DenseVector[Double] = knots(1 to n-1) - knots(0 to n-2)

      val j_ = replace(knots.length-1,x,j,knots)
      val j1, hj = j_.map( each => each.toInt - 1 )
      val j__ = j_.map(ele => if(ele == n-1) 0 else ele.toInt)
      val I = diag(DenseVector.ones[Double](n-1))

      rowProduct(matrixSubsetByIndexVector(BD,hj),pow(vectorSubsetByIndexVector(knots, j1+1)-x, 3) :/ (vectorSubsetByIndexVector(h,hj) * 6.0)) +
      rowProduct(matrixSubsetByIndexVector(BD,j__),pow(x-vectorSubsetByIndexVector(knots, j1), 3) :/ (vectorSubsetByIndexVector(h,hj)* 6.0 )) -
      rowProduct(matrixSubsetByIndexVector(BD,j1) , vectorSubsetByIndexVector(h, hj) :* (vectorSubsetByIndexVector(knots,j1+1)-x)/6.0) -
      rowProduct(matrixSubsetByIndexVector(BD,j__) , vectorSubsetByIndexVector(h, hj) :* (x-vectorSubsetByIndexVector(knots,j1))/6.0) +
      rowProduct(matrixSubsetByIndexVector(I, j1),(vectorSubsetByIndexVector(knots, j1+1)-x)/vectorSubsetByIndexVector(h,hj)) +
      rowProduct(matrixSubsetByIndexVector(I, j__),(x-vectorSubsetByIndexVector(knots, j1))/vectorSubsetByIndexVector(h,hj))

  }

  def cwrap (x0 : Double , x1 : Double, x : DenseVector[Double]): DenseVector[Double] ={
    val h  = x1-x0
    if(max(x) > x1){
      val ind = x :> x1
      x(ind.argmax to x.length-1) := x(ind.argmax to x.length-1).map(ele => x0 + (ele-x1) % h)
    }

    if(min(x) < x0){
      val ind2 = x :< x0
      x(ind2.argmax to x.length-1) := x(ind2.argmax to x.length-1).map(ele => x0 - (x0-ele) % h)
    }
    x
  }

  def replace( i : Int, x : DenseVector[Double], j: DenseVector[Double], knots : DenseVector[Double]) : DenseVector[Double] = i match {
    case 1 =>
      DenseVector.tabulate(x.length) { index => if (x(index) <= knots(i)) i else j(index) }
    case _ =>
      replace(i - 1, x, DenseVector.tabulate(x.length) { index => if (x(index) <= knots(i)) i else j(index) }, knots)
  }

  def matrixSubsetByIndexVector(  X : DenseMatrix[Double],index_vector : DenseVector[Int] ) : DenseMatrix[Double] = {
    DenseMatrix.tabulate(index_vector.length, X(::, 1).length){case (i, j) => X(::,j).valueAt(index_vector(i))}
  }

  def vectorSubsetByIndexVector(vector : DenseVector[Double],  index_vector : DenseVector[Int]) : DenseVector[Double] = {
    DenseVector.tabulate(index_vector.length){e => vector(index_vector(e))}
  }

  def rowProduct(X : DenseMatrix[Double] , v : DenseVector[Double]) : DenseMatrix[Double] = {
    DenseMatrix.tabulate(X.rows, X.cols){case (i, j) => (X(i,j)*v(i))}
  }

}
