library(readxl)
library(tm)
library(wordcloud)
library(e1071)
library(gmodels)
library(Rcpp)

imdb_labelled <- read_excel("C:\\Users\\anjan\\Desktop\\C-uppsats\\imdb_labelled.xlsx")

View(imdb_labelled)
head(imdb_labelled)
imdb_labelled$score <- factor(imdb_labelled$score)

#Check the counts of positive and negative scores
table(imdb_labelled$score)

#Create a corpus from the sentences
imdb_corpus <- VCorpus(VectorSource(imdb_labelled$review))

#Create a document-term sparse matrix directly from the corpus
imdb_dtm <- DocumentTermMatrix(imdb_corpus, control = list(
  tolower=TRUE,
  removeNumbers=TRUE,
  stopwords=TRUE,
  removePunctuation=TRUE,
  stemming=TRUE
))

#Creating training and test data
imdb_dtm_train <- imdb_dtm[1:799,]  
imdb_dtm_test <- imdb_dtm[800:1000,]

#Save labels

imdb_train_labels <-imdb_labelled[1:799,]$score
imdb_test_labels <- imdb_labelled[800:1000,]$score

#Check that the proportion of spam is similar
prop.table(table(imdb_train_labels))
prop.table(table(imdb_test_labels))

#Sampling
rm(imdb_dtm_train)
rm(imdb_dtm_test)
rm(imdb_train_labels)
rm(imdb_test_labels)

# Create random samples
set.seed(123)
train_index <- sample(1000, 799)

imdb_train <- imdb_labelled[train_index, ]
imdb_test  <- imdb_labelled[-train_index, ]

# check the proportion of class variable
prop.table(table(imdb_train$score))


train_corpus <- VCorpus(VectorSource(imdb_train$review))
test_corpus <- VCorpus(VectorSource(imdb_test$review))


#Subset the training data into spam and ham groups


positive <- subset(imdb_train, score == 1)
negative <- subset(imdb_train, score == 0)

wordcloud(train_corpus, min.freq = 20, random.order = FALSE)


wordcloud(positive$review, max.words = 50, scale = c(3,0.5),
          colors = brewer.pal(2, "Dark2"))
wordcloud(negative$review, max.words = 50, scale = c(3,0.5),
          colors = brewer.pal(2, "Dark2"))

train_dtm <- DocumentTermMatrix(train_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

test_dtm <- DocumentTermMatrix(test_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

train_dtm

test_dtm


convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

#apply() convert_counts() to columns of train/test data
train_dtm_binary <- apply(train_dtm, MARGIN = 2, convert_counts)
test_dtm_binary  <- apply(test_dtm, MARGIN = 2, convert_counts)


#Training a model on the data
imdb_classifier <- naiveBayes(as.matrix(train_dtm_binary), imdb_train$score)

#Evaluationg model performance
imdb_test_pred <- predict(imdb_classifier, as.matrix(test_dtm_binary))
head(imdb_test_pred)


CrossTable(imdb_test_pred, imdb_test$score,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

#Improving model performance #Laplace smoothing (1)
imdb_classifier2 <- naiveBayes(as.matrix(train_dtm_binary), imdb_train$score, laplace = 1)

imdb_test_pred2 <- predict(imdb_classifier2, as.matrix(test_dtm_binary))

CrossTable(imdb_test_pred2, imdb_test$score,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))







