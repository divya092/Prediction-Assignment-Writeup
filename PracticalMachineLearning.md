Prediction Assignment Writeup
=============================

I. Overview
===========

This is the report of the Peer Assessment project from Coursera’s course
- Practical Machine Learning, as part of the Specialization in Data
Science. The main goal of the project is to predict the manner in which
6 participants performed some exercise as described below. This is the
“classe” variable in the training set. The machine learning algorithm
described here is applied to the 20 test cases available in the test
data and the predictions are submitted in appropriate format to the
Course Project Prediction Quiz for automated grading.

II. Data Loading and Exploratory Analysis
=========================================

a) Dataset information
----------------------

The data for this project come from
<http://groupware.les.inf.puc-rio.br/har>.

A short description of the datasets content from the authors’ website:

“Six young health participants were asked to perform one set of 10
repetitions of the Unilateral Dumbbell Biceps Curl in five different
fashions: exactly according to the specification (Class A), throwing the
elbows to the front (Class B), lifting the dumbbell only halfway (Class
C), lowering the dumbbell only halfway (Class D) and throwing the hips
to the front (Class E).

Class A corresponds to the specified execution of the exercise, while
the other 4 classes correspond to common mistakes. Participants were
supervised by an experienced weight lifter to make sure the execution
complied to the manner they were supposed to simulate. The exercises
were performed by six male participants aged between 20-28 years, with
little weight lifting experience. We made sure that all participants
could easily simulate the mistakes in a safe and controlled manner by
using a relatively light dumbbell (1.25kg)."

b) Environment preparation
--------------------------

    library(knitr)
    library(caret)
    library(rpart)
    library(rpart.plot)
    library(rattle)
    library(randomForest)
    library(corrplot)
    library(e1071)
    library(gbm)
    set.seed(123)

c) Data Loading and Cleaning
----------------------------

The training dataset is then partinioned in 2 to create a Training set
(70% of the data) for the modeling process and a Test set (with the
remaining 30%) for the validations. The testing dataset is not changed
and will only be used for the quiz results generation.

    # Download the dataset from URL provided
    Train <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    Test  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

    # Read the downloaded data files
    training <- read.csv(url(Train))
    testing  <- read.csv(url(Test))

    # create a partition with the training dataset 
    train_part  <- createDataPartition(training$classe, p=0.7, list=FALSE)
    TrainSet <- training[train_part, ]
    TestSet  <- training[-train_part, ]
    dim(TrainSet)

    ## [1] 13737   160

The training data set is made of 19622 observations on 160 columns. We
can notice that many columns have NA values or blank values on almost
every observation. So we will remove them, because they will not produce
any information. The first seven columns give information about the
people who did the test, and also timestamps. We will not take them in
our model.

    # Here we get the indexes of the columns having at least 90% of NA or blank values on the training dataset
    indColToRemove <- which(colSums(is.na(TrainSet) |TrainSet=="")>0.9*dim(TrainSet)[1]) 
    TrainSetClean <- TrainSet[,-indColToRemove]
    TrainSetClean <- TrainSetClean[,-c(1:7)]
    dim(TrainSetClean)

    ## [1] 13737    53

    # We do the same for the test set
    indColToRemove <- which(colSums(is.na(TestSet) |TestSet=="")>0.9*dim(TestSet)[1]) 
    TestSetClean <- TestSet[,-indColToRemove]
    TestSetClean <- TestSetClean[,-1]
    dim(TestSetClean)

    ## [1] 5885   59

d) Correlation Analysis
-----------------------

A correlation among variables is analysed before proceeding to the
modeling procedures.

    corMatrix <- cor(TrainSetClean[, -53])
    corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
             tl.cex = 0.8, tl.col = rgb(0, 0, 0))

![](PracticalMachineLearning_files/figure-markdown_strict/unnamed-chunk-5-1.png)

III. Prediction Model Building
==============================

Three methods will be applied to model the regressions using the Train
dataset and the best one with higher accuracy when applied to the Test
dataset will be used for the quiz predictions. The methods are: Random
Forests, Decision Tree and Generalized Boosted Model. A Confusion Matrix
is plotted at the end of each analysis to better visualize the accuracy
of the models.

Method 1 : Random Forest
------------------------

    set.seed(123)
    controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
    modFitRandForest <- train(classe ~ ., data=TrainSetClean, method="rf",
                              trControl=controlRF)
    modFitRandForest$finalModel

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.68%
    ## Confusion matrix:
    ##      A    B    C    D    E class.error
    ## A 3901    3    0    0    2 0.001280082
    ## B   21 2629    7    1    0 0.010910459
    ## C    0   12 2380    4    0 0.006677796
    ## D    0    0   27 2222    3 0.013321492
    ## E    0    1    5    7 2512 0.005148515

    # prediction on Test dataset
    predictRandForest <- predict(modFitRandForest, newdata=TestSetClean)
    confMatRandForest <- confusionMatrix(predictRandForest, TestSetClean$classe)
    confMatRandForest

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1671    6    0    0    0
    ##          B    3 1130    8    0    0
    ##          C    0    3 1015   16    2
    ##          D    0    0    3  947    1
    ##          E    0    0    0    1 1079
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9927          
    ##                  95% CI : (0.9902, 0.9947)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9908          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9982   0.9921   0.9893   0.9824   0.9972
    ## Specificity            0.9986   0.9977   0.9957   0.9992   0.9998
    ## Pos Pred Value         0.9964   0.9904   0.9797   0.9958   0.9991
    ## Neg Pred Value         0.9993   0.9981   0.9977   0.9966   0.9994
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2839   0.1920   0.1725   0.1609   0.1833
    ## Detection Prevalence   0.2850   0.1939   0.1760   0.1616   0.1835
    ## Balanced Accuracy      0.9984   0.9949   0.9925   0.9908   0.9985

    # plot matrix results
    plot(confMatRandForest$table, col = confMatRandForest$byClass, 
         main = paste("Random Forest - Accuracy =",
                      round(confMatRandForest$overall['Accuracy'], 4)))

![](PracticalMachineLearning_files/figure-markdown_strict/unnamed-chunk-8-1.png)

Method 2 : Decision Trees
-------------------------

    set.seed(123)
    modFitDecTree <- rpart(classe ~ ., data=TrainSetClean, method="class")

    # prediction on Test dataset
    predictDecTree <- predict(modFitDecTree, newdata=TestSetClean, type="class")
    confMatDecTree <- confusionMatrix(predictDecTree, TestSetClean$classe)
    confMatDecTree

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1424  184   52   56   25
    ##          B   26  667   54   48   70
    ##          C   33  107  701   53   52
    ##          D  167  106  193  735  157
    ##          E   24   75   26   72  778
    ## 
    ## Overall Statistics
    ##                                         
    ##                Accuracy : 0.7315        
    ##                  95% CI : (0.72, 0.7428)
    ##     No Information Rate : 0.2845        
    ##     P-Value [Acc > NIR] : < 2.2e-16     
    ##                                         
    ##                   Kappa : 0.6606        
    ##                                         
    ##  Mcnemar's Test P-Value : < 2.2e-16     
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8507   0.5856   0.6832   0.7624   0.7190
    ## Specificity            0.9247   0.9583   0.9496   0.8734   0.9590
    ## Pos Pred Value         0.8179   0.7711   0.7410   0.5412   0.7979
    ## Neg Pred Value         0.9397   0.9060   0.9342   0.9494   0.9381
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2420   0.1133   0.1191   0.1249   0.1322
    ## Detection Prevalence   0.2958   0.1470   0.1607   0.2308   0.1657
    ## Balanced Accuracy      0.8877   0.7719   0.8164   0.8179   0.8390

    # plot matrix results
    plot(confMatDecTree$table, col = confMatDecTree$byClass, 
         main = paste("Decision Tree - Accuracy =",
                      round(confMatDecTree$overall['Accuracy'], 4)))

![](PracticalMachineLearning_files/figure-markdown_strict/unnamed-chunk-11-1.png)

Method 3 : Generalized Boosted Model
------------------------------------

    set.seed(123)
    controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
    modFitGBM  <- train(classe ~ ., data=TrainSetClean, method = "gbm",
                        trControl = controlGBM, verbose = FALSE)

    # prediction on Test dataset
    predictGBM <- predict(modFitGBM, newdata=TestSetClean)
    confMatGBM <- confusionMatrix(predictGBM, TestSetClean$classe)
    confMatGBM

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1646   50    0    0    5
    ##          B   18 1057   33    0   10
    ##          C    8   30  979   43    7
    ##          D    2    2   12  916   14
    ##          E    0    0    2    5 1046
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.959          
    ##                  95% CI : (0.9537, 0.964)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9482         
    ##                                          
    ##  Mcnemar's Test P-Value : 1.902e-10      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9833   0.9280   0.9542   0.9502   0.9667
    ## Specificity            0.9869   0.9871   0.9819   0.9939   0.9985
    ## Pos Pred Value         0.9677   0.9454   0.9175   0.9683   0.9934
    ## Neg Pred Value         0.9933   0.9828   0.9902   0.9903   0.9925
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2797   0.1796   0.1664   0.1556   0.1777
    ## Detection Prevalence   0.2890   0.1900   0.1813   0.1607   0.1789
    ## Balanced Accuracy      0.9851   0.9576   0.9680   0.9721   0.9826

    # plot matrix results
    plot(confMatGBM$table, col = confMatGBM$byClass, 
         main = paste("GBM - Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))

![](PracticalMachineLearning_files/figure-markdown_strict/unnamed-chunk-14-1.png)

IV. Using the Selected Model to the Test Data
=============================================

The accuracy of the 3 regression modeling methods used above and
accuracies are:

Random Forest : 0.9927 Decision Tree : 0.7315 GBM : 0.959 Based on the
results obtained, Random Forest model will be applied to predict the 20
quiz results on testing dataset as shown below.

    predictTEST <- predict(modFitRandForest, newdata=testing)
    predictTEST

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
