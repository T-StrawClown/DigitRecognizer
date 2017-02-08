par(mfrow = c(10, 20), mar = c(0,0,0,0))
x <- 5*(1:28)
y <- 5*(1:28)
img <- sample(1:nrow(ds.train), 100)
for (i in img) {
  m <- matrix(unlist(matrix(raw.train[i,-1], 28, 28, F)), 28, 28)
  #m <- apply(m, 2, rev)
  m <- t(apply(m, 1, rev))
  image(x, y, m, col = grey((255:0)/255), axes = F)
  image(x, y, m, col = grey((1:0)), axes = F)
}
matrix(raw.train[img,1], 10, 10, T)

