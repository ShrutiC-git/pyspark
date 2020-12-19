#importing important libraries
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from numpy import array

# Boilerplate Spark stuff:
#we are using a SparkContext instead of a SparkSession. SparkSession is a more conducive approacha s it allows us to use dataframes, which are essentially like a pandas dataframe
conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
sc = SparkContext(conf = conf)

#reading our csv into an RDD
rdd = sc.textFile('diabetes.csv')
rdd.collect() #collect data from the worker nodes

#removing the first line of the csv, which is the header
header = rdd.first()
rdd = rdd.filter(lambda x: x!=header)
#rdd.collect()

#changing into a list
to_csv = rdd.map(lambda x: x.split(","))
to_csv.collect()

#splitting into train,test
train,test = to_csv.randomSplit([0.8,0.2],seed=42)

#converting the data to be optimised to be fed into our Decision Tree classfier
training_data = train.map(lambda x: LabeledPoint(x[8],array([x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]])))
training_data.collect()

#extracting the labels
test_labels = test.map(lambda x: float(x[8]))
test_labels.collect()

model = DecisionTree.trainClassifier(training_data, numClasses=2,categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=6, maxBins=40)

test_results = model.predict(test)
#print('Diabetic Predictions:')
#results = test_results.collect()
#for result in results:
#    print(result)

# We can also print out the decision tree itself:
print('Learned classification tree model:')
print(model.toDebugString())

#zipping the labels from test data (test_y) and the predictions made on test_X
labelsAndPredictions = test_labels.zip(test_results)
labelsAndPredictions.collect()

#Mean squared error
testMSE = labelsAndPredictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() /\
            float(test_labels.count())
#we can do better
testMSE

predict_raw = sc.textFile("diabetes_predict.csv")
predict_data = predict_raw.map(lambda x: x.split(","))
predict_data.collect()

predictions = model.predict(predict_data)
print('Diabetic Predictions:')
results = predictions.collect()
for result in results:
    print(result)