library(randomForest)                                                          
data(iris)

var.share <- function(rf.obj, members) {
  count <- table(rf.obj$forest$bestvar)[-1]
  names(count) <- names(rf.obj$forest$ncat)
  share <- count[members] / sum(count[members])
  return(share)
}
group.importance <- function(rf.obj, groups) {
  var.imp <- as.matrix(sapply(groups, function(g) {
    sum(importance(rf.obj, 2)[g, ]*var.share(rf.obj, g))
  }))
  colnames(var.imp) <- "MeanDecreaseGini"
  return(var.imp)
}

rf.obj <- randomForest(Species ~ ., data=iris)

groups <- list(Sepal=c("Sepal.Width", "Sepal.Length"), 
               Petal=c("Petal.Width", "Petal.Length"))

group.importance(rf.obj, groups)



rf.obj <- df %>% select(age, quantity, payment_type, water_quality, waterpoint_type, district_code, region_code, status_group) %>%
  randomForest(status_group ~ ., data = .)

X_train <- df %>% select(age, quantity, payment_type, water_quality, waterpoint_type, district_code, region_code)
train.obj <-train(X_train, df$status_group)
