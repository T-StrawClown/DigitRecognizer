require(doParallel)
cluster <- makeCluster(detectCores())
registerDoParallel()

tr_control = trainControl(method = "cv",
                          number = 6)

# random forest, 6 fold cross-validation
require(randomForest)
grid.rf <- expand.grid(mtry = c(2^(1:4)))
mdl.rf <- train(label ~ ., data = ds.train, method = "rf", trControl = tr_control, tuneGrid = grid.rf)
# mdl.rf <- randomForest(classe ~ ., data = ds.training, mtry = 2, ntree = 200)

stopCluster()

# grid.rf <- expand.grid(mtry = c(2^(5:6)))
# mdl.rf1 <- train(label ~ ., data = ds.train, method = "rf", trControl = tr_control, tuneGrid = grid.rf)
# 
pred.rf <- predict(mdl.rf$finalModel, newdata = ds.val)
conf.rf <- confusionMatrix(round(pred.rf), ds.val$label)

