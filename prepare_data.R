raw.train <- read.csv("data/train.csv")
raw.test <- read.csv("data/test.csv")

require(dplyr)
raw.train.zero_cols <- data.frame(names = names(raw.train[colSums(raw.train) == 0]), stringsAsFactors = F)
raw.test.zero_cols <- data.frame(names = names(raw.test[colSums(raw.test) == 0]), stringsAsFactors = F)

ds.zero_cols <- inner_join(raw.train.zero_cols, raw.test.zero_cols, c("names" = "names"))
ds.train.all <- raw.train %>%
  select(-which(names(raw.train) %in% ds.zero_cols$names))
ds.train.all$label <- as.factor(ds.train.all$label)
ds.test <- raw.test %>%
  select(-which(names(raw.test) %in% ds.zero_cols$names))

set.seed(42)
require(caret)
inTrain <- createDataPartition(y = ds.train.all$label, p = .7, list = F)
ds.train <- ds.train.all[inTrain,]
ds.val <- ds.train.all[-inTrain,]

