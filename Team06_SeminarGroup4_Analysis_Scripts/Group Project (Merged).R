setwd("C:/Users/Andy/Desktop/BC2407 Course Materials/Group Project")

# Packages used:  --------------------------------------------------------------------
{
  library(corrplot)
  library(fastDummies)
  library(randomForest)
  library(caret)
  library(caTools)
  library(smotefamily)
  library(rpart)
  library(rpart.plot)
  library(car)
  library(ggplot2)
  library(clustMixType)
  library(tibble)
  library(pROC)
  library(cluster)
  library(proxy)
}

# Data Processing (Churn Dataset) --------------------------------------------------------------------
{
  ecommerce <- read.csv("E Commerce Dataset.csv",  stringsAsFactors = T)
  summary(ecommerce)
  
  # Count total NAs
  total_nas <- sum(is.na(ecommerce))
  total_nas
  
  # Summary of the dataset after conversion
  summary(ecommerce)
  
  # Remove customer_id
  ecommerce <- subset(ecommerce, select = -CustomerID)
  
  # Replace blanks with NA
  ecommerce[ecommerce == ""] <- NA
  
  #==== Imputation
  # Identify columns with missing values
  missing_cols <- c("Tenure", "WarehouseToHome", "HourSpendOnApp", "OrderAmountHikeFromlastYear", 
                    "CouponUsed", "OrderCount", "DaySinceLastOrder")
  
  # Impute missing values using random forest imputation with rough fix (median)
  ecommerce_imputed <- rfImpute(x = ecommerce[, missing_cols],
                                y = ecommerce$Churn,  # Assuming "Churn" is the response variable
                                na.roughfix = TRUE,  # Perform rough fix imputation
                                median.impute = TRUE,  # Use median imputation for rough fix
                                importance = TRUE)  # Optional: to obtain variable importance measures
  
  # Summary of the dataset after imputation
  summary(ecommerce_imputed)
  
  # Remove the "Churn" column from ecommerce_imputed
  ecommerce_imputed <- ecommerce_imputed[, -which(names(ecommerce_imputed) == "ecommerce$Churn")]
  
  # Replace missing values in the original dataset with imputed values
  ecommerce[, missing_cols] <- ecommerce_imputed
  
  # Summary of the dataset after replacing missing values
  summary(ecommerce)
  
  
  # Convert column to a factor
  ecommerce$PreferredLoginDevice <- factor(ecommerce$PreferredLoginDevice)
  ecommerce$CityTier <- factor(ecommerce$CityTier)
  ecommerce$PreferredPaymentMode <- factor(ecommerce$PreferredPaymentMode)
  ecommerce$Gender <- factor(ecommerce$Gender)
  ecommerce$PreferedOrderCat <- factor(ecommerce$PreferedOrderCat)
  ecommerce$SatisfactionScore <- factor(ecommerce$SatisfactionScore)
  ecommerce$MaritalStatus <- factor(ecommerce$MaritalStatus)
  ecommerce$Complain <- factor(ecommerce$Complain)
  
  # Check the levels of categorical variables
  levels(ecommerce$Churn)
  levels(ecommerce$PreferredLoginDevice)
  levels(ecommerce$CityTier)
  levels(ecommerce$PreferredPaymentMode)
  levels(ecommerce$Gender)
  levels(ecommerce$PreferedOrderCat)
  levels(ecommerce$SatisfactionScore)
  levels(ecommerce$MaritalStatus)
  levels(ecommerce$Complain)
  
  #==== Consistency
  # Replacement
  ecommerce$PreferredPaymentMode[ecommerce$PreferredPaymentMode == "CC"] <- "Credit Card"
  ecommerce$PreferredPaymentMode[ecommerce$PreferredPaymentMode == "COD"] <- "Cash on Delivery"
  ecommerce$PreferedOrderCat[ecommerce$PreferedOrderCat == "Mobile"] <- "Mobile Phone"
  ecommerce$PreferredLoginDevice[ecommerce$PreferredLoginDevice == "Phone"] <- "Mobile Phone"
  
  # Renaming values
  ecommerce$Churn <- ifelse(ecommerce$Churn == 0, "No", "Yes")
  ecommerce$Complain <- ifelse(ecommerce$Complain == 0, "No", "Yes")
  
  write.csv(ecommerce, file = "Users/Andy/Desktop/BC2407 Course Materials/Group Project/E Commerce Dataset (imputed).csv", row.names = FALSE)
}

# Data Processing --------------------------------------------------------------------
{
  churn.df <- read.csv("Final.csv",  stringsAsFactors = T)
  summary(churn.df)
  
  sum(is.na(churn.df))
  sum(duplicated(churn.df))
  
  print(sapply(churn.df, class))
  
  
  churn.df$VisitorType <- NULL
  churn.df$CustomerID <- NULL
  
  # Encoding the Month column
  {
    churn.df$Month <- factor(churn.df$Month, levels = c("Jan", "Feb", "Mar", "Apr", "May", "June",
                                                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))
    month_to_num <- c('Jan' = 1, 'Feb' = 2, 'Mar' = 3, 'Apr' = 4, 'May' = 5,
                      'June' = 6, 'Jul' = 7, 'Aug' = 8, 'Sep' = 9, 
                      'Oct' = 10, 'Nov' = 11, 'Dec' = 12)
    churn.df$Month <- sapply(churn.df$Month, function(x) month_to_num[x])
    churn.df$Month <- as.factor(churn.df$Month)
  }
  
  # Converting categorical columns
  {````
    columns <- c("Churn", "PreferredLoginDevice", "CityTier", "PreferredPaymentMode", "Gender",
                 "PreferedOrderCat", "SatisfactionScore", "MaritalStatus", "Complain",
                 "OperatingSystems", "Browser", "Region", "TrafficType")
    
    for (col in columns) {
      churn.df[[col]] <- factor(churn.df[[col]])
    }
    
    churn.df$Weekend <- as.numeric(churn.df$Weekend)
    churn.df$Weekend <- as.factor(churn.df$Weekend)
  }
  
  numerical_cols <- names(churn.df[sapply(churn.df, is.numeric)])
  categorical_cols <- names(churn.df[sapply(churn.df, is.factor)])
  
  # Outlier Detection
  for(column in numerical_cols) {
    Q1 <- quantile(churn.df[[column]], 0.25, na.rm = TRUE)
    Q3 <- quantile(churn.df[[column]], 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    
    number <- sum(churn.df[[column]] < (Q1 - 1.5 * IQR) | churn.df[[column]] > (Q3 + 1.5 * IQR), na.rm = TRUE)
    
    cat(column, ":", number, "\n")
  }
  
  set.seed(123)
  train <- sample.split(Y = churn.df$Churn, SplitRatio = 0.7)
  trainset <- subset(churn.df, train == T)
  testset <- subset(churn.df, train == F)
}

# Random Forest (Original) --------------------------------------------------------------------
{
  
  set.seed(123)
  
  rf_model <- randomForest(Churn ~ ., data=trainset, importance = T)
  rf_model
  plot(rf_model)
  rf_model$predicted
  rf_model$err.rate
  
  var.impt <- importance(rf_model)
  varImpPlot(rf_model, type = 1)
  ranked_importance <- var.impt[order(var.impt[, "MeanDecreaseAccuracy"], decreasing = TRUE), ]
  print(ranked_importance)
  
  rf_predictions <- predict(rf_model, newdata = testset)
  
  # Create a confusion matrix
  confusion_matrix <- confusionMatrix(rf_predictions, testset$Churn, mode = "everything")
  
  # Print the confusion matrix
  print(confusion_matrix)
  
}

# CART (Original) --------------------------------------------------------------------
{
  set.seed(123)
  
  cart1 <- rpart(Churn ~ ., data = trainset, method = "class",
                 control = rpart.control(minsplit = 2, cp = 0))
  printcp(cart1)
  plotcp(cart1)
  # rpart.plot(cart1, nn = T, main = "Max Tree in marun_sample2")
  
  # Extracting the Optimal Tree
  CVerror.cap <- cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xerror"] + cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xstd"]
  
  # Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree cart1.
  i <- 1; j<- 4
  while (cart1$cptable[i,j] > CVerror.cap) {
    i <- i + 1
  }
  
  print(i)
  
  # Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
  cp.opt = ifelse(i > 1, sqrt(cart1$cptable[i,1] * cart1$cptable[i-1,1]), 1)
  
  # Prune the max tree using a particular CP value
  cart2 <- prune(cart1, cp = cp.opt)
  printcp(cart2, digits = 3)
  
  # Visualising the tree
  print(cart2)
  rpart.plot(cart2, nn = T, main = "Optimal Tree in marun_sample2")
  
  cart2$variable.importance
  summary(cart2)
  
  cart_predictions <- predict(cart2, newdata = testset, type = "class")
  confusion_matrix <- confusionMatrix(cart_predictions, testset$Churn, mode = "everything")
  print(confusion_matrix)
  
}

# Logistic Regression (Original) --------------------------------------------------------------------
{
  set.seed(123)
  
  log_model1 <- glm(Churn ~ . , family = binomial, data = trainset)
  summary(log_model1)
  
  # Akaike Information Criterion
  log_model2 <- step(log_model1)
  summary(log_model2)
  
  # Checking VIF
  vif(log_model2)
  
  # Remove PreferedOrderCat and CashbackAmount from the model as it has the highest VIF
  log_model3 <- glm(formula = Churn ~ Tenure + PreferredLoginDevice + CityTier + 
                      WarehouseToHome + PreferredPaymentMode + Gender + NumberOfDeviceRegistered + 
                      SatisfactionScore + MaritalStatus + NumberOfAddress + 
                      Complain + OrderCount + DaySinceLastOrder + 
                      ProductRelated + ExitRates + PageValues + SpecialDay + Month, 
                    family = binomial, data = trainset)
  vif(log_model3)
  summary(log_model3)
  
  # Calculate Odds Ratios and Confidence Intervals
  OR <- exp(coef(log_model3))
  OR.CI <- exp(confint(log_model3))
  
  print(OR)
  print(OR.CI)
  
  log_predictions <- predict(log_model3, newdata=testset, type='response')
  y.hat <- ifelse(log_predictions > 0.5, 1, 0)
  
  # Create a confusion matrix
  confusion_matrix <- confusionMatrix(as.factor(y.hat), as.factor(testset$Churn), mode = "everything")
  print(confusion_matrix)
  
}

# Feature Selection (Predictive Model) -----------------------------------------------------------------
{
  columns_to_keep <- c("Churn", "Tenure", "Complain", "DaySinceLastOrder", "Administrative_Duration", "Informational_Duration",
                       "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "Month", "TrafficType")
  model1.churn.df <- churn.df[, columns_to_keep]
  
  set.seed(123)
  train <- sample.split(Y = model1.churn.df$Churn, SplitRatio = 0.7)
  trainset <- subset(model1.churn.df, train == T)
  testset <- subset(model1.churn.df, train == F)
  
  str(trainset$TrafficType)
  str(testset$TrafficType)
  testset <- subset(testset, TrafficType != "18")
  
}

# Random Forest (Before Oversampling) --------------------------------------------------------------------
{
  
  set.seed(123)
  
  rf_model <- randomForest(Churn ~ ., data=trainset, importance = T)
  rf_model
  plot(rf_model)
  rf_model$predicted
  rf_model$err.rate
  
  var.impt <- importance(rf_model)
  varImpPlot(rf_model, type = 1)
  ranked_importance <- var.impt[order(var.impt[, "MeanDecreaseAccuracy"], decreasing = TRUE), ]
  print(ranked_importance)
  
  rf_predictions <- predict(rf_model, newdata = testset)
  
  # Create a confusion matrix
  confusion_matrix <- confusionMatrix(rf_predictions, testset$Churn, mode = "everything")
  
  # Print the confusion matrix
  print(confusion_matrix)
  
}

# CART (Before Oversampling) --------------------------------------------------------------------
{
  set.seed(123)
  
  cart1 <- rpart(Churn ~ ., data = trainset, method = "class",
                 control = rpart.control(minsplit = 2, cp = 0))
  printcp(cart1)
  plotcp(cart1)
  # rpart.plot(cart1, nn = T, main = "Max Tree in marun_sample2")
  
  # Extracting the Optimal Tree
  CVerror.cap <- cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xerror"] + cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xstd"]
  
  # Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree cart1.
  i <- 1; j<- 4
  while (cart1$cptable[i,j] > CVerror.cap) {
    i <- i + 1
  }
  
  print(i)
  
  # Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
  cp.opt = ifelse(i > 1, sqrt(cart1$cptable[i,1] * cart1$cptable[i-1,1]), 1)
  
  # Prune the max tree using a particular CP value
  cart2 <- prune(cart1, cp = cp.opt)
  printcp(cart2, digits = 3)
  
  # Visualising the tree
  print(cart2)
  rpart.plot(cart2, nn = T, main = "Optimal Tree in marun_sample2")
  
  cart2$variable.importance
  summary(cart2)
  
  cart_predictions <- predict(cart2, newdata = testset, type = "class")
  confusion_matrix <- confusionMatrix(cart_predictions, testset$Churn, mode = "everything")
  print(confusion_matrix)
  
}

# Logistic Regression (Before Oversampling) --------------------------------------------------------------------
{
  set.seed(123)
  
  log_model1 <- glm(Churn ~ . , family = binomial, data = trainset)
  summary(log_model1)
  
  # Calculate Odds Ratios and Confidence Intervals
  OR <- exp(coef(log_model1))
  OR.CI <- exp(confint(log_model1))
  
  print(OR)
  print(OR.CI)
  
  log_predictions <- predict(log_model1, newdata=testset, type='response')
  y.hat <- ifelse(log_predictions > 0.5, 1, 0)
  
  # Create a confusion matrix
  confusion_matrix <- confusionMatrix(as.factor(y.hat), as.factor(testset$Churn), mode = "everything")
  print(confusion_matrix)
  
}

# Oversampling -----------------------------------------------------------------
{
  print(table(trainset$Churn))
  # 0    1 
  # 3277  664 
  
  summary(trainset)
  
  df_balanced <- trainset
  
  # Performing oversampling using smotelibrary
  smotetest <- df_balanced
  
  # Converting columns to numeric
  smotetest[] <- lapply(smotetest, function(x) if(is.factor(x)) as.numeric(as.character(x)) else as.numeric(x))
  print(sapply(smotetest,class))
  
  set.seed(123)
  dup_size = sum(smotetest$Churn == 0)/sum(smotetest$Churn == 1)
  smotetest_oversampled = SMOTE(smotetest[,-17],target=smotetest$Churn, K = 3, dup_size = dup_size-1)
  
  View(smotetest_oversampled)
  
  df_balanced <- smotetest_oversampled$data
  
  # Rename the "class" column to "Churn" and recode the values
  df_balanced$Churn <- ifelse(df_balanced$class == 1, 1, 0)
  df_balanced$Churn <- factor(df_balanced$Churn, levels = c("1","0"))
  
  # Remove the original "class" column
  df_balanced$class <- NULL
  
  # Rounding the values after doing oversampling
  columns <- c("Complain", "TrafficType", "Month")
  for (col in columns) {
    df_balanced[[col]] <- round(df_balanced[[col]])
  }
  
  # Converting back into factor
  for (col in columns) {
    df_balanced[[col]] <- factor(df_balanced[[col]])
  }
  
  print(sapply(df_balanced, class))
  summary(df_balanced)
  
  testset$Month <- factor(testset$Month, levels = levels(df_balanced$Month))
  testset$TrafficType <- factor(testset$TrafficType, levels = levels(df_balanced$TrafficType))
  
}

# Random Forest (After Oversampling) --------------------------------------------------------------------
{
  
  set.seed(123)
  
  rf_model <- randomForest(Churn ~ ., data=df_balanced, importance = T)
  rf_model
  plot(rf_model)
  rf_model$predicted
  rf_model$err.rate
  
  var.impt <- importance(rf_model)
  varImpPlot(rf_model, type = 1)
  
  rf_predictions <- predict(rf_model, newdata = testset)
  
  # Create a confusion matrix
  confusion_matrix <- confusionMatrix(rf_predictions, testset$Churn, mode = "everything")
  print(confusion_matrix)
  
}

# CART (After Oversampling) --------------------------------------------------------------------
{
  set.seed(123)
  
  cart1 <- rpart(Churn ~ ., data = df_balanced, method = "class",
                 control = rpart.control(minsplit = 2, cp = 0))
  printcp(cart1)
  plotcp(cart1)
  # rpart.plot(cart1, nn = T, main = "Max Tree in marun_sample2")
  
  # Extracting the Optimal Tree
  CVerror.cap <- cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xerror"] + cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xstd"]
  
  # Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree cart1.
  i <- 1; j<- 4
  while (cart1$cptable[i,j] > CVerror.cap) {
    i <- i + 1
  }
  
  print(i)
  
  # Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
  cp.opt = ifelse(i > 1, sqrt(cart1$cptable[i,1] * cart1$cptable[i-1,1]), 1)
  
  # Prune the max tree using a particular CP value
  cart2 <- prune(cart1, cp = cp.opt)
  printcp(cart2, digits = 3)
  
  # Visualising the tree
  print(cart2)
  rpart.plot(cart2, nn = T, main = "Optimal Tree in marun_sample2")
  
  cart2$variable.importance
  summary(cart2)
  
  cart_predictions <- predict(cart2, newdata = testset, type = "class")
  confusion_matrix <- confusionMatrix(cart_predictions, testset$Churn, mode = "everything")
  print(confusion_matrix)
  
}

# Logistic Regression (After Oversampling) --------------------------------------------------------------------
{
  set.seed(123)
  
  log_model1 <- glm(Churn ~ . , family = binomial, data = df_balanced)
  summary(log_model1)
  
  # Calculate Odds Ratios and Confidence Intervals
  OR <- exp(coef(log_model1))
  OR.CI <- exp(confint(log_model1))
  
  print(OR)
  print(OR.CI)
  
  log_predictions <- predict(log_model1, newdata=testset, type='response')
  y.hat <- ifelse(log_predictions < 0.5, 1, 0)
  
  # Create a confusion matrix
  confusion_matrix <- confusionMatrix(as.factor(y.hat), as.factor(testset$Churn), mode = "everything")
  print(confusion_matrix)
  
}

# Random Forest (Evaluation) --------------------------------------------------------------------
{
  
  set.seed(123)
  
  rf_model <- randomForest(Churn ~ ., data=df_balanced, importance = T)
  rf_model
  plot(rf_model)
  
  oob_predictions <- rf_model$votes[,2]
  roc_obj <- roc(df_balanced$Churn, oob_predictions)
  plot(roc_obj, main="ROC Curve", col="#1c61b6")
  auc(roc_obj)
  
}

# Feature Selection (Clustering Model) -----------------------------------------------------------------
{
  columns_to_keep <- c("Churn", "PreferedOrderCat", "PreferredLoginDevice", "CityTier", "WarehouseToHome", "PreferredPaymentMode", 
                       "MaritalStatus", "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount")
  model2.churn.df <- churn.df[, columns_to_keep]
  
  model2.churn.df <- model2.churn.df[model2.churn.df$Churn == "1", ]
  model2.churn.df$Churn <- NULL
  
}

# K-Prototypes --------------------------------------------------------------------
{
  set.seed(123)
  
  n <- 10
  wss <- numeric(n)
  
  # Loop over possible values of k
  for (k in 1:n) {
    model <- kproto(x = model2.churn.df, k = k, lambda = 1, iter.max = 50, nstart = 20)
    wss[k] <- model$tot.withinss
  }
  print(wss)
  
  wss_df <- tibble(clusters = 1:n, wss = wss)
  
  scree_plot <- ggplot(wss_df, aes(x = clusters, y = wss, group = 1)) +
    geom_point(size = 4)+
    geom_line() +
    scale_x_continuous(breaks = c(2, 4, 6, 8, 10)) +
    xlab('Number of clusters')
  
  scree_plot +
    geom_hline(
      yintercept = wss, 
      linetype = 'dashed', 
      col = c(rep('#000000'))
    )
  
  # Build model with optimal k clusters
  set.seed(123)
  k <- 4
  optimal_model <- kproto(x =model2.churn.df, k = k, lambda = 1, iter.max = 50, nstart = 20)
  
  # Assign the cluster membership to original dataframe
  #model2.churn.df$Cluster <- optimal_model$cluster
  #model2.churn.df$Cluster <- as.factor(model2.churn.df$Cluster)
  
  centers <- optimal_model$centers
  print(centers)
  centers$Cluster <- as.factor(1:4)
  
  # write.csv(centers, file = "Cluster Centres.csv", row.names = FALSE)
  
}
