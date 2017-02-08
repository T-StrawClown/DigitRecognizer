require(neuralnet)
require(nnet)

ds.train$label <- as.integer(ds.train$label)
ds.val$label <- as.integer(ds.val$label)
ds.train_cc <- data.frame(cbind(class.ind(ds.train$label), ds.train[,-1]))
#names(ds.train_cc)[1:10] <- 1:10
ds.val_cc <- data.frame(cbind(class.ind(ds.val$label), ds.val[,-1]))
#names(ds.val_cc)[1:10] <- 1:10

x.names <- names(ds.train)[-1]
formula <- as.formula(paste('X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 ~ ', paste(x.names, collapse = ' + ')))

mdl.nn <- neuralnet(formula, data = ds.train_cc, algorithm = 'backprop', hidden = c(100),
                    lifesign = 'full', lifesign.step = 1, rep = 1, stepmax = 100,
                    learningrate = 25, linear.output = F, err.fct = 'ce')
mdl.pred <- prediction(mdl.nn)
pred.nn <- compute(mdl.nn, ds.val_cc[,-(1:10)], rep = 10)
