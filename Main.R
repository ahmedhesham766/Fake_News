# Step 1: Prepare the dataset
# Assuming you have a CSV file with two columns: "text" (news content) and "label" (fake or real)
setwd("D:/big data/project")
news <- read.csv("news.csv")
news$label[news$label == 'FAKE'] <- 0
news$label[news$label == 'REAL'] <- 1
dataset <- news[,3:4]
remove(news)
######################################################################
# Step 2: Preprocess the text
# You may need to install and load additional packages like 'tm' for text preprocessing
library(tm)
library(tokenizers)
library(text2vec)
library(ggplot2)
library(glmnet)    #logistic regression
library(readr)
library(dplyr)
library(caTools)   #split
# define preprocessing function and tokenization function
prep_fun = tolower
tok_fun = word_tokenizer
# Remove nulls
dataset <- na.omit(dataset)
# Filter the dataset to include only labels 0 and 1.
dataset <- filter(dataset, dataset$label == 1 | dataset$label == 0) 
# Remove mentions, urls, emojis, numbers, punctuations, etc.
dataset$text <- gsub("@\\w+", "", dataset$text)
dataset$text <- gsub("https?://.+", "", dataset$text)
dataset$text <- gsub("\\d+\\w*\\d*", "", dataset$text)
dataset$text <- gsub("#\\w+", "", dataset$text)
dataset$text <- gsub("[^\x01-\x7F]", "", dataset$text)
dataset$text <- gsub("[[:punct:]]", " ", dataset$text)
# Remove spaces and newlines
dataset$text <- gsub("\n", " ", dataset$text)
dataset$text <- gsub("^\\s+", "", dataset$text)
dataset$text <- gsub("\\s+$", "", dataset$text)
dataset$text <- gsub("[ |\t]+", " ", dataset$text)
# Apply preprocessing functions: lowercase the text, remove English stopwords, remove numbers, and stem the text using the "english" stemmer.
dataset$text <- prep_fun(dataset$text)
dataset$text <- removeWords(dataset$text, stopwords("en"))
dataset$text <- removeNumbers(dataset$text)
dataset$text <- stemDocument(dataset$text,"english")

######################################################################
# Step 3: Split the dataset
#Set the random seed
set.seed(123)
split <- sample.split(dataset$label, SplitRatio = 0.7)
train_data <- subset(dataset, split == TRUE)
test_data <- subset(dataset, split == FALSE)

######################################################################
# Step 4: Tokenize and create a document-term matrix (DTM)
# Tokenize the text in the training data using the 'word_tokenizer' function
it_train = itoken(train_data$text, 
                  tokenizer = word_tokenizer, 
                  ids = train_data$label)
# Create a vocabulary from the tokenized training data.
vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
#Create a document-term matrix (DTM) for the training data using the vectorizer.
dtm_train = create_dtm(it_train, vectorizer)
dim(dtm_train)

######################################################################
#Step 5: Train the logistic regression model
glmnet_classifier = cv.glmnet(x = dtm_train, y = train_data$label, 
                                  family = 'binomial', 
                                  alpha = 1,
                                  type.measure = "auc",
                                  nfolds = 4,
                                  thresh = 1e-4,
                                  maxit =  1e3)
######################################################################
#Step 6: Visualize the model
plot(glmnet_classifier)
######################################################################
#Step 7: Prepare the test data
#it_test = tok_fun(test_data$text)
# Tokenize the text in the test data using the 'word_tokenizer' function
it_test <- tok_fun(test_data$text)
# turn off progressbar because it won't look nice in rmd
# Transform the tokenized test data into an iterator using the 'itoken' function
it_test = itoken(it_test, ids = test_data$label)

dtm_test = create_dtm(it_test, vectorizer)

test_data$label <- as.numeric(test_data$label)
preds = predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test_data$label, preds)

predicted = data.frame(Class = round(preds,0), Article= 1:1895)

library(caret)
cm <- confusionMatrix(data =factor(predicted$Class), reference =factor(test_data$label))
cm

library(ROCR)
pred <- prediction(preds, test_data$label)
pred
perf <- performance(pred,"tpr","fpr")
plot(perf, colorize = TRUE, main = "ROC Curve")
# Pie chart
pie(table(dataset$label), col = c("red", "green"), main = "Label Distribution")

library(wordcloud)
# Visualize the most frequent words in the dataset. This can provide insights into the common terms used in fake and real news.
wordcloud(dataset$text, scale=c(4, 0.5), max.words=100, random.order=FALSE)
# Compare the distribution of text lengths between fake and real news. This can help identify potential differences in text length between the two classes
text_length <- nchar(dataset$text)
boxplot(text_length ~ label, data = dataset, ylab = "text_length", xlab = "Label")
# Confusion Matrix Heatmap
library(ggplot2)
library(gplots)

conf_mat <- table(predicted$Class, test_data$label)

heatmap(conf_mat, col = c("red", "green"), main = "Confusion Matrix",
        labRow = rowSums(conf_mat), labCol = colSums(conf_mat))



