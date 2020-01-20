import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

val data = sc.textFile("/home/rohith/Desktop/ratings.dat");
val ratings = data.map(_.split("::") match { case Array(userid, movieid, rate,ts) =>
  Rating(userid.toInt, movieid.toInt, rate.toDouble)
});
val Array(training_data, test_data) = ratings.randomSplit(Array(0.6, 0.4))
val rank = 10;
val numIterations = 10;
val model = ALS.train(training_data, rank, numIterations, 0.01);
val usersProducts = test_data.map { case Rating(user, product, rate) =>(user, product)};
val predictions = model.predict(usersProducts).map { case Rating(user, product, rate) =>((user, product), rate)}
val ratesAndPreds = test_data.map { case Rating(user, product, rate) =>((user, product), rate)}.join(predictions)
val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
  val err = (r1 - r2)
  err * err
}.mean()
println("Mean Squared Error = " + MSE)