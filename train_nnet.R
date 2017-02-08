require(nnet)

mdl.nn <- nnet(formula = label ~ ., data = ds.train, size = 36, MaxNWts = 60000)
pred.nn <- predict(mdl.nn, ds.val, "class")
conf.nn <- confusionMatrix(pred.nn, ds.val$label)

ds.train_sign <- ds.train
ds.train_sign[,2:ncol(ds.train_sign)] <- sign(ds.train_sign[,2:ncol(ds.train_sign)])
ds.val_sign <- ds.val
ds.val_sign[,2:ncol(ds.val_sign)] <- sign(ds.val_sign[,2:ncol(ds.val_sign)])


mdl.nn_sign <- nnet(formula = label ~ ., data = ds.train_sign, size = 36, MaxNWts = 60000)
pred.nn_sign <- predict(mdl.nn_sign, ds.val_sign, "class")
conf.nn_sign <- confusionMatrix(pred.nn_sign, ds.val_sign$label)
