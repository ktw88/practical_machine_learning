Practical Machine Learning
========================================================

Wearable computing is one of hottest fields nowadays as it allows the collection
of large amount of data rather inexpensively. However, due to raw nature of the
data from the device, it is necessary to process them before it is possible to
make some some predictions about the activity that an individual is carrying out.

Here, we have a dataset of 

# Reading in of Data


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
require(randomForest)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
# Read in the data
pml.training<-read.csv("pml-training.csv",header=T)
pml.testing<-read.csv("pml-testing.csv",header=T)
```

# Data Cleaning

We will first perform some data checking on the training dataset.

```r
table(apply(is.na(pml.training),2,sum))
```

```
## 
##     0 19216 
##    93    67
```

```r
nrow(pml.training)
```

```
## [1] 19622
```
From the above, we notice that 67 columns have 19216 NA values while the rest of the 93
columns do not have 0 values. 

This indicates that 19216/19622 of the rows in some of the columns have missing information
and they will not provide us much valuable information in doing our prediction. As such, we 
will remove the redundant columns and retainonly those columns which contain the information
we want.



```r
# Extract columns which have fewer than 1000 "NA" values
clean.index<-which(apply(is.na(pml.training),2,sum) < 1000)
pml.training.clean<-pml.training[, clean.index]
pml.testing.clean<-pml.testing[, clean.index]
```

We can now look at the summary of the clean data.


```r
summary(pml.training.clean)
```

```
##        X            user_name    raw_timestamp_part_1 raw_timestamp_part_2
##  Min.   :    1   adelmo  :3892   Min.   :1.32e+09     Min.   :   294      
##  1st Qu.: 4906   carlitos:3112   1st Qu.:1.32e+09     1st Qu.:252912      
##  Median : 9812   charles :3536   Median :1.32e+09     Median :496380      
##  Mean   : 9812   eurico  :3070   Mean   :1.32e+09     Mean   :500656      
##  3rd Qu.:14717   jeremy  :3402   3rd Qu.:1.32e+09     3rd Qu.:751891      
##  Max.   :19622   pedro   :2610   Max.   :1.32e+09     Max.   :998801      
##                                                                           
##           cvtd_timestamp  new_window    num_window    roll_belt    
##  28/11/2011 14:14: 1498   no :19216   Min.   :  1   Min.   :-28.9  
##  05/12/2011 11:24: 1497   yes:  406   1st Qu.:222   1st Qu.:  1.1  
##  30/11/2011 17:11: 1440               Median :424   Median :113.0  
##  05/12/2011 11:25: 1425               Mean   :431   Mean   : 64.4  
##  02/12/2011 14:57: 1380               3rd Qu.:644   3rd Qu.:123.0  
##  02/12/2011 13:34: 1375               Max.   :864   Max.   :162.0  
##  (Other)         :11007                                            
##    pitch_belt        yaw_belt      total_accel_belt kurtosis_roll_belt
##  Min.   :-55.80   Min.   :-180.0   Min.   : 0.0              :19216   
##  1st Qu.:  1.76   1st Qu.: -88.3   1st Qu.: 3.0     #DIV/0!  :   10   
##  Median :  5.28   Median : -13.0   Median :17.0     -1.908453:    2   
##  Mean   :  0.31   Mean   : -11.2   Mean   :11.3     0.000673 :    1   
##  3rd Qu.: 14.90   3rd Qu.:  12.9   3rd Qu.:18.0     0.005503 :    1   
##  Max.   : 60.30   Max.   : 179.0   Max.   :29.0     -0.016850:    1   
##                                                     (Other)  :  391   
##  kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
##           :19216            :19216              :19216   
##  #DIV/0!  :   32     #DIV/0!:  406     #DIV/0!  :    9   
##  47.000000:    4                       0.000000 :    4   
##  -0.150950:    3                       0.422463 :    2   
##  -0.684748:    3                       0.000748 :    1   
##  11.094417:    3                       -0.003095:    1   
##  (Other)  :  361                       (Other)  :  389   
##  skewness_roll_belt.1 skewness_yaw_belt  max_yaw_belt    min_yaw_belt  
##           :19216             :19216            :19216          :19216  
##  #DIV/0!  :   32      #DIV/0!:  406     -1.1   :   30   -1.1   :   30  
##  0.000000 :    4                        -1.4   :   29   -1.4   :   29  
##  -2.156553:    3                        -1.2   :   26   -1.2   :   26  
##  -3.072669:    3                        -0.9   :   24   -0.9   :   24  
##  -6.324555:    3                        -1.3   :   22   -1.3   :   22  
##  (Other)  :  361                        (Other):  275   (Other):  275  
##  amplitude_yaw_belt  gyros_belt_x      gyros_belt_y      gyros_belt_z   
##         :19216      Min.   :-1.0400   Min.   :-0.6400   Min.   :-1.460  
##  0.00   :   12      1st Qu.:-0.0300   1st Qu.: 0.0000   1st Qu.:-0.200  
##  0.0000 :  384      Median : 0.0300   Median : 0.0200   Median :-0.100  
##  #DIV/0!:   10      Mean   :-0.0056   Mean   : 0.0396   Mean   :-0.131  
##                     3rd Qu.: 0.1100   3rd Qu.: 0.1100   3rd Qu.:-0.020  
##                     Max.   : 2.2200   Max.   : 0.6400   Max.   : 1.620  
##                                                                         
##   accel_belt_x      accel_belt_y    accel_belt_z    magnet_belt_x  
##  Min.   :-120.00   Min.   :-69.0   Min.   :-275.0   Min.   :-52.0  
##  1st Qu.: -21.00   1st Qu.:  3.0   1st Qu.:-162.0   1st Qu.:  9.0  
##  Median : -15.00   Median : 35.0   Median :-152.0   Median : 35.0  
##  Mean   :  -5.59   Mean   : 30.1   Mean   : -72.6   Mean   : 55.6  
##  3rd Qu.:  -5.00   3rd Qu.: 61.0   3rd Qu.:  27.0   3rd Qu.: 59.0  
##  Max.   :  85.00   Max.   :164.0   Max.   : 105.0   Max.   :485.0  
##                                                                    
##  magnet_belt_y magnet_belt_z     roll_arm        pitch_arm     
##  Min.   :354   Min.   :-623   Min.   :-180.0   Min.   :-88.80  
##  1st Qu.:581   1st Qu.:-375   1st Qu.: -31.8   1st Qu.:-25.90  
##  Median :601   Median :-320   Median :   0.0   Median :  0.00  
##  Mean   :594   Mean   :-346   Mean   :  17.8   Mean   : -4.61  
##  3rd Qu.:610   3rd Qu.:-306   3rd Qu.:  77.3   3rd Qu.: 11.20  
##  Max.   :673   Max.   : 293   Max.   : 180.0   Max.   : 88.50  
##                                                                
##     yaw_arm        total_accel_arm  gyros_arm_x      gyros_arm_y    
##  Min.   :-180.00   Min.   : 1.0    Min.   :-6.370   Min.   :-3.440  
##  1st Qu.: -43.10   1st Qu.:17.0    1st Qu.:-1.330   1st Qu.:-0.800  
##  Median :   0.00   Median :27.0    Median : 0.080   Median :-0.240  
##  Mean   :  -0.62   Mean   :25.5    Mean   : 0.043   Mean   :-0.257  
##  3rd Qu.:  45.88   3rd Qu.:33.0    3rd Qu.: 1.570   3rd Qu.: 0.140  
##  Max.   : 180.00   Max.   :66.0    Max.   : 4.870   Max.   : 2.840  
##                                                                     
##   gyros_arm_z     accel_arm_x      accel_arm_y      accel_arm_z    
##  Min.   :-2.33   Min.   :-404.0   Min.   :-318.0   Min.   :-636.0  
##  1st Qu.:-0.07   1st Qu.:-242.0   1st Qu.: -54.0   1st Qu.:-143.0  
##  Median : 0.23   Median : -44.0   Median :  14.0   Median : -47.0  
##  Mean   : 0.27   Mean   : -60.2   Mean   :  32.6   Mean   : -71.2  
##  3rd Qu.: 0.72   3rd Qu.:  84.0   3rd Qu.: 139.0   3rd Qu.:  23.0  
##  Max.   : 3.02   Max.   : 437.0   Max.   : 308.0   Max.   : 292.0  
##                                                                    
##   magnet_arm_x   magnet_arm_y   magnet_arm_z  kurtosis_roll_arm
##  Min.   :-584   Min.   :-392   Min.   :-597           :19216   
##  1st Qu.:-300   1st Qu.:  -9   1st Qu.: 131   #DIV/0! :   78   
##  Median : 289   Median : 202   Median : 444   0.01388 :    1   
##  Mean   : 192   Mean   : 157   Mean   : 306   0.01574 :    1   
##  3rd Qu.: 637   3rd Qu.: 323   3rd Qu.: 545   0.01619 :    1   
##  Max.   : 782   Max.   : 583   Max.   : 694   -0.02438:    1   
##                                               (Other) :  324   
##  kurtosis_picth_arm kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm
##          :19216             :19216           :19216            :19216    
##  #DIV/0! :   80     #DIV/0! :   11   #DIV/0! :   77    #DIV/0! :   80    
##  -0.00484:    1     0.55844 :    2   -0.00051:    1    0.00000 :    1    
##  0.00981 :    1     0.65132 :    2   0.00445 :    1    -0.00184:    1    
##  -0.01311:    1     -0.01548:    1   0.00494 :    1    0.00189 :    1    
##  -0.02967:    1     -0.01749:    1   0.00646 :    1    0.00708 :    1    
##  (Other) :  322     (Other) :  389   (Other) :  325    (Other) :  322    
##  skewness_yaw_arm roll_dumbbell    pitch_dumbbell    yaw_dumbbell    
##          :19216   Min.   :-153.7   Min.   :-149.6   Min.   :-150.87  
##  #DIV/0! :   11   1st Qu.: -18.5   1st Qu.: -40.9   1st Qu.: -77.64  
##  0.55053 :    2   Median :  48.2   Median : -21.0   Median :  -3.32  
##  -1.62032:    2   Mean   :  23.8   Mean   : -10.8   Mean   :   1.67  
##  0.00000 :    1   3rd Qu.:  67.6   3rd Qu.:  17.5   3rd Qu.:  79.64  
##  -0.00311:    1   Max.   : 153.6   Max.   : 149.4   Max.   : 154.95  
##  (Other) :  389                                                      
##  kurtosis_roll_dumbbell kurtosis_picth_dumbbell kurtosis_yaw_dumbbell
##         :19216                 :19216                  :19216        
##  #DIV/0!:    5          -0.5464:    2           #DIV/0!:  406        
##  -0.2583:    2          -0.9334:    2                                
##  -0.3705:    2          -2.0833:    2                                
##  -0.5855:    2          -2.0851:    2                                
##  -2.0851:    2          -2.0889:    2                                
##  (Other):  393          (Other):  396                                
##  skewness_roll_dumbbell skewness_pitch_dumbbell skewness_yaw_dumbbell
##         :19216                 :19216                  :19216        
##  #DIV/0!:    4          0.1090 :    2           #DIV/0!:  406        
##  0.1110 :    2          -0.2328:    2                                
##  -0.9324:    2          -0.3521:    2                                
##  1.0312 :    2          -0.7036:    2                                
##  0.0011 :    1          1.0326 :    2                                
##  (Other):  395          (Other):  396                                
##  max_yaw_dumbbell min_yaw_dumbbell amplitude_yaw_dumbbell
##         :19216           :19216           :19216         
##  -0.6   :   20    -0.6   :   20    0.00   :  401         
##  0.2    :   19    0.2    :   19    #DIV/0!:    5         
##  -0.8   :   18    -0.8   :   18                          
##  -0.3   :   16    -0.3   :   16                          
##  0.0    :   15    0.0    :   15                          
##  (Other):  318    (Other):  318                          
##  total_accel_dumbbell gyros_dumbbell_x  gyros_dumbbell_y gyros_dumbbell_z
##  Min.   : 0.0         Min.   :-204.00   Min.   :-2.10    Min.   : -2.4   
##  1st Qu.: 4.0         1st Qu.:  -0.03   1st Qu.:-0.14    1st Qu.: -0.3   
##  Median :10.0         Median :   0.13   Median : 0.03    Median : -0.1   
##  Mean   :13.7         Mean   :   0.16   Mean   : 0.05    Mean   : -0.1   
##  3rd Qu.:19.0         3rd Qu.:   0.35   3rd Qu.: 0.21    3rd Qu.:  0.0   
##  Max.   :58.0         Max.   :   2.22   Max.   :52.00    Max.   :317.0   
##                                                                          
##  accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x
##  Min.   :-419.0   Min.   :-189.0   Min.   :-334.0   Min.   :-643     
##  1st Qu.: -50.0   1st Qu.:  -8.0   1st Qu.:-142.0   1st Qu.:-535     
##  Median :  -8.0   Median :  41.5   Median :  -1.0   Median :-479     
##  Mean   : -28.6   Mean   :  52.6   Mean   : -38.3   Mean   :-328     
##  3rd Qu.:  11.0   3rd Qu.: 111.0   3rd Qu.:  38.0   3rd Qu.:-304     
##  Max.   : 235.0   Max.   : 315.0   Max.   : 318.0   Max.   : 592     
##                                                                      
##  magnet_dumbbell_y magnet_dumbbell_z  roll_forearm     pitch_forearm   
##  Min.   :-3600     Min.   :-262.0    Min.   :-180.00   Min.   :-72.50  
##  1st Qu.:  231     1st Qu.: -45.0    1st Qu.:  -0.74   1st Qu.:  0.00  
##  Median :  311     Median :  13.0    Median :  21.70   Median :  9.24  
##  Mean   :  221     Mean   :  46.1    Mean   :  33.83   Mean   : 10.71  
##  3rd Qu.:  390     3rd Qu.:  95.0    3rd Qu.: 140.00   3rd Qu.: 28.40  
##  Max.   :  633     Max.   : 452.0    Max.   : 180.00   Max.   : 89.80  
##                                                                        
##   yaw_forearm     kurtosis_roll_forearm kurtosis_picth_forearm
##  Min.   :-180.0          :19216                :19216         
##  1st Qu.: -68.6   #DIV/0!:   84         #DIV/0!:   85         
##  Median :   0.0   -0.8079:    2         0.0012 :    1         
##  Mean   :  19.2   -0.9169:    2         -0.0073:    1         
##  3rd Qu.: 110.0   0.0128 :    1         0.0249 :    1         
##  Max.   : 180.0   -0.0227:    1         0.0354 :    1         
##                   (Other):  316         (Other):  317         
##  kurtosis_yaw_forearm skewness_roll_forearm skewness_pitch_forearm
##         :19216               :19216                :19216         
##  #DIV/0!:  406        #DIV/0!:   83         #DIV/0!:   85         
##                       -0.1912:    2         0.0000 :    4         
##                       -0.4126:    2         -0.6992:    2         
##                       -0.0004:    1         -0.0113:    1         
##                       -0.0013:    1         -0.0131:    1         
##                       (Other):  317         (Other):  313         
##  skewness_yaw_forearm max_yaw_forearm min_yaw_forearm
##         :19216               :19216          :19216  
##  #DIV/0!:  406        #DIV/0!:   84   #DIV/0!:   84  
##                       -1.2   :   32   -1.2   :   32  
##                       -1.3   :   31   -1.3   :   31  
##                       -1.4   :   24   -1.4   :   24  
##                       -1.5   :   24   -1.5   :   24  
##                       (Other):  211   (Other):  211  
##  amplitude_yaw_forearm total_accel_forearm gyros_forearm_x  
##         :19216         Min.   :  0.0       Min.   :-22.000  
##  0.00   :  322         1st Qu.: 29.0       1st Qu.: -0.220  
##  #DIV/0!:   84         Median : 36.0       Median :  0.050  
##                        Mean   : 34.7       Mean   :  0.158  
##                        3rd Qu.: 41.0       3rd Qu.:  0.560  
##                        Max.   :108.0       Max.   :  3.970  
##                                                             
##  gyros_forearm_y  gyros_forearm_z  accel_forearm_x  accel_forearm_y
##  Min.   : -7.02   Min.   : -8.09   Min.   :-498.0   Min.   :-632   
##  1st Qu.: -1.46   1st Qu.: -0.18   1st Qu.:-178.0   1st Qu.:  57   
##  Median :  0.03   Median :  0.08   Median : -57.0   Median : 201   
##  Mean   :  0.08   Mean   :  0.15   Mean   : -61.7   Mean   : 164   
##  3rd Qu.:  1.62   3rd Qu.:  0.49   3rd Qu.:  76.0   3rd Qu.: 312   
##  Max.   :311.00   Max.   :231.00   Max.   : 477.0   Max.   : 923   
##                                                                    
##  accel_forearm_z  magnet_forearm_x magnet_forearm_y magnet_forearm_z
##  Min.   :-446.0   Min.   :-1280    Min.   :-896     Min.   :-973    
##  1st Qu.:-182.0   1st Qu.: -616    1st Qu.:   2     1st Qu.: 191    
##  Median : -39.0   Median : -378    Median : 591     Median : 511    
##  Mean   : -55.3   Mean   : -313    Mean   : 380     Mean   : 394    
##  3rd Qu.:  26.0   3rd Qu.:  -73    3rd Qu.: 737     3rd Qu.: 653    
##  Max.   : 291.0   Max.   :  672    Max.   :1480     Max.   :1090    
##                                                                     
##  classe  
##  A:5580  
##  B:3797  
##  C:3422  
##  D:3216  
##  E:3607  
##          
## 
```

Based on the summary, we can also notice that alot of fields
are basically empty and we will remove these columns as well.


```r
emptyindex<-apply(pml.training.clean, 2, function(x){
  sum(x == "") > 1000
  })

pml.training.cleaner<-pml.training.clean[,!emptyindex]
pml.testing.cleaner<-pml.testing.clean[, !emptyindex]
```

From the summary, we also note that the first 6 columns do not provide very
descriptive information about the row, we will drop these columns as well.


```r
# Remove the first 6 columns because they carry not very useful information
pml.training.cleaner<-pml.training.cleaner[,-c(1:6)]
pml.testing.cleaner<-pml.testing.cleaner[,-c(1:6)]
```

# Create Data partition

Now, we will will split the training data we have into two sets, a training set
and a testing set. We will retain 70% in the training set for model building and 
use 30% for testing

```r
inTrain<-createDataPartition(pml.training.cleaner$classe, p=0.7, list=FALSE)
training<-pml.training.cleaner[inTrain,]
testing<-pml.training.cleaner[-inTrain,]
```

# Random Forest Models

## Cross Validation
To estimate the in sample accuracy, we will make use of the K-fold cross validation
method (specifically, 4-fold). 


```r
# We will perform 4-fold cross validation here
set.seed(12345)
foldnum<-4
folds<-createFolds(y=training$classe, k=foldnum, list=TRUE, returnTrain=TRUE)

models_all<-list()
crossval_results<-c()

for (i in 1:foldnum){
  training.itrain<-training[folds[[i]],]
  training.itest<-training[-folds[[i]],]
  model_i <- randomForest(classe~.,data=training.itrain)
  models_all[[i]]<-model_i
  
  crossvalidation_results<-confusionMatrix(predict(model_i, training.itest), training.itest$classe)$overall
  crossval_results<-rbind(crossval_results, crossvalidation_results)
}

# Results of the different models we have
crossval_results
```

```
##                         Accuracy  Kappa AccuracyLower AccuracyUpper
## crossvalidation_results   0.9953 0.9941        0.9924        0.9973
## crossvalidation_results   0.9965 0.9956        0.9939        0.9982
## crossvalidation_results   0.9936 0.9919        0.9903        0.9960
## crossvalidation_results   0.9965 0.9956        0.9939        0.9982
##                         AccuracyNull AccuracyPValue McnemarPValue
## crossvalidation_results       0.2842              0           NaN
## crossvalidation_results       0.2843              0           NaN
## crossvalidation_results       0.2843              0           NaN
## crossvalidation_results       0.2845              0           NaN
```

From the above, we can have a look at teh accuracy of each cross validation models. The accuracy of all models
seem to be fairly high and robust.

## Model Selection

Using 4-fold cross validation, we would have 4 different models. We would select the model that gave the highest
accuracy as our best model.


```r
# Select the model which can 
bestmod_index<-which(crossval_results[,1] == max(crossval_results[,1]))
bestmodel<-models_all[[bestmod_index]]
```


# In-sample Error

Now, since we have our best model, we can test it once again to see the in-sample
error of our model.


```r
confusionMatrix(predict(bestmodel, training), training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    1    0    0    0
##          B    0 2657    3    0    0
##          C    0    0 2392    6    0
##          D    0    0    1 2246    4
##          E    0    0    0    0 2521
## 
## Overall Statistics
##                                         
##                Accuracy : 0.999         
##                  95% CI : (0.998, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.999         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    0.998    0.997    0.998
## Specificity             1.000    1.000    0.999    1.000    1.000
## Pos Pred Value          1.000    0.999    0.997    0.998    1.000
## Neg Pred Value          1.000    1.000    1.000    0.999    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.175    0.164    0.184
## Balanced Accuracy       1.000    1.000    0.999    0.998    0.999
```

As can be seen, our model works rather well on our training data

# Out-of-sample Error

We can also estimate how well our model works on the test dataset.


```r
confusionMatrix(predict(bestmodel, testing), testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    5    0    0
##          C    0    0 1020    4    0
##          D    0    0    1  960    1
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.997, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.998         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    0.994    0.996    0.999
## Specificity             1.000    0.999    0.999    1.000    1.000
## Pos Pred Value          1.000    0.996    0.996    0.998    1.000
## Neg Pred Value          1.000    1.000    0.999    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.173    0.163    0.184
## Detection Prevalence    0.284    0.194    0.174    0.163    0.184
## Balanced Accuracy       1.000    0.999    0.997    0.998    1.000
```

As seen above, our model works very well on the test dataset.


# Prediction of Activity

With the model we have built, we can then try to predict the activity in the pml.test set.


```r
results<-predict(bestmodel, pml.testing.cleaner)
results
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

# Writing of files to text


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


pml_write_files(as.character(results))
```
