# Sentiment Analysis: Comparing VADER and Roberta Models

This project explores sentiment analysis techniques applied to Amazon reviews, focusing on two models: VADER (Valence Aware Dictionary for sEntiment Reasoning) and Roberta. The goal is to compare their performance, identify discrepancies, and uncover insights about their strengths and weaknesses.

---

## Key Insights from VADER and Roberta

### **Why Compare VADER and Roberta?**
VADER and Roberta represent two fundamentally different approaches to sentiment analysis:
- **VADER**: A rule-based, lexical approach that uses a sentiment dictionary to assign scores to words.
- **Roberta**: A transformer-based neural network model that gathers contextual understanding through deep learning.

### **Comparison of Results**

**Neutral Label Bias**:
   - VADER often assigns higher confidence to positive or negative labels due to its reliance on predefined word valence scores.
   - Roberta, in contrast, is more conservative and frequently assigns the "neutral" label because it evaluates sentiment based on context rather than isolated words.

**So What?**
- Simplicity vs Contextual Understanding: VADER is lightweight and quick, suitable for applications where a basic understanding of sentiment suffices. Roberta excels in nuanced, context-heavy scenarios but requires more computational resources.

![ScreenShot](https://github.com/umbertomaglione/SentimentAnalysis_AmazonReviews/blob/main/pairplot.png "VADER vs Roberta")


### **Error Diagnostics**:
*"I felt energized within five minutes, but it lasted for about 45 minutes. I paid $3.99 for this drink. I could have just drunk a cup of coffee and saved my money."*

- VADER (Misclassification): The word "energized" influenced VADER’s positive sentiment score despite the negative overall tone.
- Roberta: Correctly identifies a neutral/negative tone from the context.
   
*"This was sooooo delicious but too bad I ate them too fast and gained 2 lbs! My fault."*
- Both models misinterpret this sentence. VADER focuses on "too bad" because it lacks context-awareness, while Roberta’s training data may not include enough examples of nuanced language to grasp the playful tone. 


---

## Model Evaluation: Performance Metrics

### **Accuracy and F-Score**
Three additional machine learning models were applied for comparison using sklearn: Linear SVM, Logistic Regression, and Decision Tree. The dataset was vectorized using the TF-IDF method to weigh less frequent but more informative words.
A Grid Search was conducted to optimize parameters for the Linear SVM model, particularly the kernel type and regularization (C-value)

| Model               | Accuracy | F-Score (Positive) |
|---------------------|----------|---------------------|
| Linear SVM          | 0.79     | 0.88                |
| Logistic Regression | 0.79     | 0.88                |
| Decision Tree       | 0.74     | 0.84                |

**Key Takeaways:**
- Linear SVM and Logistic Regression outperformed the Decision Tree model in terms of both accuracy and F-score.
- The Decision Tree’s performance suffered from sensitivity to noisy and imbalanced data.

### **Impact of Dataset Imbalance**
The original dataset had a bias toward 5-star reviews, reflected in poor F-scores for negative and neutral sentiment across all models. Balancing the dataset led to significant improvements in capturing minority classes, particularly for the Decision Tree model, which showed a trade-off between positive and negative accuracy.

![ScreenShot](https://github.com/umbertomaglione/SentimentAnalysis_AmazonReviews/blob/main/stars.png "Stars")

---
