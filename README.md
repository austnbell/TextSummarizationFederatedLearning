# TextSummarizationFederatedLearning
Text Summarization Tool developed with Federated Learning

For full information, read the complete blog post found here:

### Summary:
Develop text summarization tool where distributed data is incorporated through federated learning. Federated Learning allows training on distributed datasets and guarantees data privacy through sharing model parameters rather than data throughout training process. 

Text Summarizer implemented with a SummaRuNNer model. Paper is located in "Papers" directory or the arxiv folder is found [here](https://arxiv.org/pdf/1611.04230.pdf)


### Analysis:
Simulate a real-world scenario where an AI software vendor wants to sell a text summarization to two external buyers.  The data that will be summarizer is considered highly confidential, so the buyers do not want to allow the AI vendor to access data.  Federated learning enables collaborative training without sharing data. 

Model flow:
* Extract Daily Mail Summarization data
* Split data across Vendor and Buyers by topic
* Develop extractive summarization Model
 * Federated learning guarantees that data is not shared
* Compare Results

1. Topic Model (/Programs/TopicModelling)
  * Creates a topic model using the 20newsgroup dataset
  * Splits Daily Mail data according to topic 
  * Each party receives a subset of the topics (no party overlaps topics)

2. SummaRunner (/programs/SummaRunner)
 * Prep data for input (inputs whole document split by sentences)
  * Convert data to embedding idx 
  * Create Vocab Object 
 * SummaRunner Model

SummaRuNNer architecture
![alt text](https://github.com/austnbell/TextSummarizationFederatedLearning/blob/master/Papers/SummaRuNNer_Architecture.png)


3. Evaluation (/programs/Evaluation)
 * Evaluator class
  * Convert idx to text
  * Extract gold and predicted summaries
  * Compute Rouge Scores

### Results

Metrics:
![alt_text](https://github.com/austnbell/TextSummarizationFederatedLearning/blob/master/Papers/evaluation_table.png)

Example of Extractive Summary:
![alt_text](https://github.com/austnbell/TextSummarizationFederatedLearning/blob/master/Papers/Summary_example.png)
Red highlights sentences that appear in one summary, but not the other.
