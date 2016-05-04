import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import java.io._


object HelloWorld {
  def main(args: Array[String]): Unit = {

    //val y_vector = DenseVector[Double]((0.0 to 1.0 by .05).toArray)
    //val x_vector = DenseVector[Double]((0.0 to 1.0 by .05).toArray)


    //val knots = DenseVector[Double]((.0 to 1.0 by .02).toArray)
    val knots = DenseVector[Double](0.0,0.2,0.5,0.65,0.7,0.75,0.8,0.85,0.92,0.95,0.97)
    val xp = DenseVector[Double]((.0 to 1.0 by .01).toArray)
    val size = DenseVector[Double](1.42,1.58,1.78,1.99,1.99,1.99,2.13,2.13,2.13, 2.32,2.32,2.32,2.32,2.32,2.43,2.43,2.78,2.98,2.98)
    val wear = DenseVector[Double](4.0,4.2,2.5,2.6,2.8,2.4,3.2,2.4,2.6,4.8,2.9,3.8,3.0,2.7,3.1,3.3,3.0,2.8,1.7)
    val pp = size-min(size)
    val x = pp/(max(pp))
    //val lambda = 1e-8
    val lambda = 1e-5
    val initial_norm = 1

    val girth_ =  DenseVector(8.3, 8.6, 8.8, 10.5, 10.7, 10.8, 11.0, 11.0, 11.1, 11.2, 11.3, 11.4, 11.4, 11.7, 12.0, 12.9,
      12.9, 13.3, 13.7, 13.8, 14.0, 14.2, 14.5, 16.0, 16.3, 17.3, 17.5, 17.9, 18.0, 18.0, 20.6)
    val girth = (girth_ - min(girth_))/(max(girth_)-min(girth_))

    val height_ = DenseVector[Double](70, 65, 63, 72, 81, 83, 66, 75, 80, 75, 79, 76, 76, 69, 75, 74, 85, 86, 71, 64, 78, 80, 74, 72, 77, 81, 82,
      80, 80, 80, 87)
    val height = (height_ - min(height_))/(max(height_)-min(height_))


    val volume = DenseVector(10.3, 10.3, 10.2, 16.4, 18.8, 19.7, 15.6, 18.2, 22.6, 19.9, 24.2, 21.0, 21.4, 21.3, 19.1, 22.2,
      33.8, 27.4, 25.7, 24.9, 34.5, 31.7, 36.3, 38.3, 42.6, 55.4, 55.7, 58.3, 51.5, 51.0, 77.0)

    // no knots
    val (x_matrix, s_list) = am_setup(List(height, girth), List(knots,knots))


/*/
    for (i <- 1 to 30;
        j <- 1 to 30) {
      val lambda_1 = lambda*Math.pow(2.0,i-1)
      val lambda_2 = lambda*Math.pow(2.0,j-1)

      println(s"current lambda_1 $lambda_1")
      println(s"current lambda_2 $lambda_2")

      val  (_, _, beta, gcv) = fit_gam(volume, x_matrix, s_list, DenseVector(lambda_1, lambda_2))


    }
**/
    val  (x__, y__, beta_vector, gcv) = fit_gam(volume, x_matrix, s_list, DenseVector(335.54432, 2.0E-5))





    val fitted = x__ * beta_vector
    println(exp(fitted(0 to 30)))
    breeze.linalg.csvwrite(new File("/Users/kbrusch/dev/gam-fitting-in-spark/data/fitted.csv")
      ,exp(fitted(0 to 30)).asDenseMatrix)
    /*
    println(volume)
    val grith_coeffs = beta_vector(0 to 4)
    val fitted_girth = x__ * DenseVector.vertcat(grith_coeffs ,DenseVector.zeros[Double](5))
    println(fitted_girth(0 to 30))

    val height_vector = beta_vector(5 to 9)
    val fitted_height= x__ * DenseVector.vertcat(DenseVector.zeros[Double](5),height_vector)
    println(fitted_height(0 to 30))
    //println(y__.length)




        val bd_list = getBD(knots)
        val BD = bd_list(0) \ bd_list(1)
        val X = predict_matrix_cyclic_smooth(xp ,knots, BD)
        val Xa = DenseMatrix.vertcat(X, (matrix_square_root(bd_list(1).t * BD) * sqrt(lambda)))




        breeze.linalg.csvwrite(new File("/Users/kbrusch/dev/gam-fitting-in-spark/data/predict.csv")
          ,Xa)

        breeze.linalg.csvwrite(new File("/Users/kbrusch/dev/gam-fitting-in-spark/data/am_X.csv")
          ,x_)

        breeze.linalg.csvwrite(new File("/Users/kbrusch/dev/gam-fitting-in-spark/data/am_beta.csv")
          ,beta.asDenseMatrix)

        val (a,b,c) = try_lambda(wear, x, knots, lambda, 1 ,wear.length-1, List())
        breeze.linalg.csvwrite(new File("/Users/kbrusch/dev/gam-fitting-in-spark/data/single.csv")
          ,(Xa*b).asDenseMatrix)




            val (_x, _beta, _y) = prs_fit(wear,x, knots, lambda)

        println("bd_list(0)")
        //println(bd_list(0))
        //println("bd_list(1)")
        //println(bd_list(1))

    val (x_, beta, y_) = gam(wear, List(x,x), lambdas, List(knots,knots))
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

        //println(model_coeff)

        val qqq = Xpredict * (X \ wear)
        val sss = Xpredict * model_coeff




        breeze.linalg.csvwrite(new File("/Users/kbrusch/dev/gam-fitting-in-spark/data/X.csv")
          ,qqq.asDenseMatrix )
          */

  }

  def fit_gam(y: DenseVector[Double], Xa : DenseMatrix[Double], S_list : List[DenseMatrix[Double]], lambdas : DenseVector[Double]) =  {

    val S = S_list.zipWithIndex.map{ case (s_matrix, lambda_index) => s_matrix * lambdas(lambda_index)}.foldLeft(DenseMatrix.ones[Double](S_list(0).rows,1)){(agg,ele) => DenseMatrix.horzcat(agg,ele)}
    val X1 = DenseMatrix.vertcat(Xa(::, 1 to Xa.cols-1), matrix_square_root(S(::, 1 to S.cols-1)))
    val initial_beta = DenseVector.vertcat(DenseVector[Double](1),DenseVector.ones[Double](X1.cols-1))

    gam_fitting_loop(X1, y, X1.cols,X1.rows, initial_beta, 1.0, false)
  }

  def gam_fitting_loop(X1 : DenseMatrix[Double], y : DenseVector[Double], q : Int ,n : Int , beta : DenseVector[Double], norm : Double, converged : Boolean): (DenseMatrix[Double], DenseVector[Double], DenseVector[Double], Double) = converged match {
    case true =>
      val gcv = (norm*n)/pow((n-trA(X1, y.length-1)),2)
      println(s"converged with a gcv of $gcv")
      (X1, y, beta, gcv)
    case _ =>

      val eta_ = (X1 * beta)
      val eta = eta_(0 to y.length-1)
      //println(X1)
      //println(eta)
      //println(y)
      val mu = exp(eta)

      val z_ = ((y-mu)/mu) + eta
      val z = DenseVector.vertcat(z_, DenseVector.zeros[Double](X1.rows-y.length))

      val new_beta = X1 \ z
      val tra_ = trA(X1, y.length-1)
      val new_norm = form_residuals_squared(X1, new_beta, z, y.length-1)
      val converged = !(abs(new_norm-norm) > 0.0001*norm)
      val delta = new_norm-norm
      val gcv = (norm*n)/pow((n-trA(X1, y.length-1)),2)
      println(s"not converged with gcv of $gcv and delta of $delta")

      gam_fitting_loop(X1, y, q, n ,new_beta, new_norm, converged)
  }

  def try_lambda(y : DenseVector[Double], x : DenseVector[Double], knots : DenseVector[Double], lambda : Double, i : Int, n : Int, scores : List[Any]) : (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = i match {

    case 2 =>
      val (xa, coeffs,ya) = prs_fit(y,x,knots,lambda)
      //// watch for xa
      val tra = trA(xa, n)
      val rss = form_residuals_squared(xa, coeffs, ya, n)
      val gcv = n*rss/pow((n-tra),2)
      (xa, coeffs,ya)
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


  def gam(response : DenseVector[Double], predictors : List[DenseVector[Double]], lambdas : DenseVector[Double], knots : List[DenseVector[Double]] ) = {
    val (xa_, s_) = am_setup(predictors, knots)
    fit_am(response, xa_, s_, lambdas)
  }

  def am_setup(predictors : List[DenseVector[Double]], knots : List[DenseVector[Double]]) :  (DenseMatrix[Double], List[DenseMatrix[Double]]) ={

    val Xa = predictors.zip(knots).map{ case (x_vector, knot_vector) => form_X(x_vector, knot_vector)}.foldLeft(DenseMatrix.ones[Double](predictors(0).length,1)){(agg,ele) => DenseMatrix.horzcat(agg,ele)}
    val S = knots.zipWithIndex.map{ case (knot_vector, index) => align_S(form_S(knot_vector),index, knots.length)}
    (Xa, S)
  }

  def align_S(S : DenseMatrix[Double], index : Int, max : Int) : DenseMatrix[Double] = {
      val full_S = DenseMatrix.zeros[Double](S.rows*max, S.cols)
      full_S(index*S.rows to ((index+1)*S.rows)-1, 0 to S.cols-1) := S
      full_S
  }

  def fit_am(y: DenseVector[Double], Xa : DenseMatrix[Double], S_list : List[DenseMatrix[Double]], lambdas : DenseVector[Double]) =  {

    val S = S_list.zipWithIndex.map{ case (s_matrix, lambda_index) => s_matrix * lambdas(lambda_index)}.foldLeft(DenseMatrix.ones[Double](10,1)){(agg,ele) => DenseMatrix.horzcat(agg,ele)}
    val X1 = DenseMatrix.vertcat(Xa(::, 1 to Xa.cols-1), matrix_square_root(S(::, 1 to S.cols-1)))
    val y_ = DenseVector.vertcat(y, DenseVector.zeros[Double](X1.rows-y.length))
    val beta =  X1 \ y_
    val tra_ = trA(X1, y.length-1)
    val norm  = form_residuals_squared(X1, beta, y_, y.length-1)
    val n = X1.rows
    (beta, pow(norm*n/(tra_ - n),2))
  }






  def am_fitting_loop(X1 : DenseMatrix[Double], y : DenseVector[Double], q : Int ,n : Int , beta : DenseVector[Double], norm : Double, norm_switch : Boolean): (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = norm_switch match {
    case true =>
      (X1, y, beta)
    case _ =>

      val y_ = DenseVector.vertcat(y, DenseVector.zeros[Double](X1.rows-y.length))
      val beta =  X1 \ y_
      val tra_ = trA(X1, y.length-1)
      val new_norm = form_residuals_squared(X1, beta, y_, y.length-1)
      println(!(abs(new_norm-norm) > 0.00002))
      am_fitting_loop(X1, y, q, n ,beta, new_norm, !(abs(new_norm-norm) > 0.00002))
  }








  def fit_am_(y: DenseVector[Double], Xa : DenseMatrix[Double], S_list : List[DenseMatrix[Double]], lambdas : DenseVector[Double]) : (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {

    val S = S_list.zipWithIndex.map{ case (s_matrix, lambda_index) => s_matrix :* lambdas(lambda_index)}.foldLeft(DenseMatrix.ones[Double](10,1)){(agg,ele) => DenseMatrix.horzcat(agg,ele)}
    //println(S(::, 1 to S.cols-1))
    //println(Xa(::, 1 to Xa.cols-1))
    //println(matrix_square_root(S(::, 1 to S.cols-1)))
    val X1 = DenseMatrix.vertcat(Xa(::, 1 to Xa.cols-1), matrix_square_root(S(::, 1 to S.cols-1)))

    val ya = DenseVector.vertcat(y, DenseVector.zeros[Double](X1.rows-y.length))
    (X1, X1 \ ya, ya)

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
    val hat_ = hat(X)
    val first_n_elements = hat_(0 to n)
    sum(first_n_elements)
  }


  def vcov(X : DenseMatrix[Double]) : DenseMatrix[Double] = {
    val x_mean = DenseVector.ones[Double](X.rows)*mean(X(::,*))
    val d = X - x_mean
  //  pow(X.rows.toDouble-1.0,-1.0) * d.t * d
    d
  }



  def hat_matrix(X : DenseMatrix[Double]) : DenseMatrix[Double] = {
    X*(X.t\X)*X.t
  }

  def hat_matrix_smooth(X : DenseMatrix[Double]) : DenseMatrix[Double] = {
    X*vcov(X)*X.t
  }

  def hat(X : DenseMatrix[Double]) : DenseVector[Double] = {
    val x =  qr(X)
    val n = X.rows
    val rank_ = rank(X)
    val new_matrix = pow(x.q * DenseMatrix.vertcat(diag(DenseVector.ones[Double](rank_)),DenseMatrix.zeros[Double](n-rank_ ,rank_)),2)
    sum(new_matrix(*,::))
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
    d.eigenvectors*diag(sqrt(abs(d.eigenvalues)))*d.eigenvectors.t
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
