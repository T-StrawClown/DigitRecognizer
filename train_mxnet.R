require(mxnet)

data <- mx.symbol.Variable("data")
hl1 <- mx.symbol.FullyConnected(data, name = "hl1", num_hidden = 360)
act1 <- mx.symbol.Activation(hl1, name = "relu1", act_type = "relu")

hl2 <- mx.symbol.FullyConnected(data, name = "hl2", num_hidden = 36)
act2 <- mx.symbol.Activation(hl2, name = "relu2", act_type = "relu")

hl3 <- mx.symbol.FullyConnected(act2, name = "hl3", num_hidden = 10)
sm <- mx.symbol.SoftmaxOutput(hl3, name = "sm")

#train.x <- t(data.matrix(ds.train_sign[,-1], rownames.force = F))
#train.y <- as.vector(as.integer(ds.train_sign[,1], rownames.force = F)-1)
#train.x <- t(data.matrix(sign(ds.train.all[,-1]), rownames.force = F))
#train.y <- as.vector(as.integer(ds.train.all[,1], rownames.force = F)-1)
train.x <- t(data.matrix(ds.train.all[,-1]/255, rownames.force = F))
train.y <- as.vector(as.integer(ds.train.all[,1], rownames.force = F)-1)

dev <- mx.cpu()

mx.set.seed(42)
mdl.mxnet <- mx.model.FeedForward.create(sm, X = train.x, y = train.y,
                                         ctx = dev, num.round = 20,
                                         array.batch.size = 100,
                                         learning.rate = 0.07, momentum = 0.9,
                                         eval.metric = mx.metric.accuracy,
                                         initializer = mx.init.uniform(0.07),
                                         epoch.end.callback = mx.callback.log.train.metric(100))

#ds.val.mxnet_sign <- t(data.matrix(ds.val_sign[,-1], rownames.force = F))
#pred.mxnet <- apply(predict(mdl.mxnet, ds.val.mxnet_sign), 2, which.max) - 1
#pred.mxnet <- predict(mdl.mxnet, t(data.matrix(sign(ds.test), rownames.force = F)))
pred.mxnet <- predict(mdl.mxnet, t(data.matrix(ds.test/255, rownames.force = F)))
pred.mxnet <- apply(pred.mxnet, 2, which.max) - 1

#conf.mxnet <- confusionMatrix(pred.mxnet, ds.val_sign[,1])
sub.mxnet <- data.frame(ImageId=1:nrow(ds.test), Label=pred.mxnet)
write.csv(sub.mxnet, paste0(getwd(), "/submission/mxnet_V5.csv"), row.names=FALSE, quote=FALSE)
