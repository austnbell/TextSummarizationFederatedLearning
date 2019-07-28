# TextSummarizationFederatedLearning
Text Summarization Tool developed with Federated Learning

For full information, read the complete blog post found here:

### Summary:
Compare a text summarizing tool developed using federated learning to one developed using traditional learning methods. Federated Learning allows training on distributed datasets and guarantees data privacy through sharing model parameters rather than data throughout training process. 

Text Summarizer implemented with a SummaRuNNer model. Paper is located in "Papers" directory or the arxiv folder is found [here](https://arxiv.org/pdf/1611.04230.pdf)


### Analysis:
Simulate a real-world scenario where an AI software vendor wants to sell a text summarization to two external buyers.  The data that will be summarizer is considered highly confidential, so the buyers do not want to allow the AI vendor to access data.  Federated learning enables collaborative training without sharing data. 

The goal of this analysis is to compare how effective federated learning might be in a real-world scenario such as this. i.e., Does a text summarizer trained using all parties' data perform more effectively than just a text summarizer developed on just the vendor's data? If so, what is the difference? 

Model flow:
* Extract Daily Mail Summarization data
* Split data across Vendor and Buyers by topic
* Develop two extractive summarization tools
  * Baseline Model trained using only Vendor Data
  * Federated Learning Model trained with complete set of data (without sharing data)
* Compare Results

1. Topic Model (/Programs/TopicModelling)
  * Creates a topic model using the 20newsgroup dataset
  * Splits Daily Mail data according to topic 
  * Each party receives a subset of the topics (no party overlaps topics)

2. SummaRunner (/programs/SummaRunner)
..* Prep data for input (inputs whole document split by sentences)
....* Convert data to embedding idx 
....* Create Vocab Object 
..* SummaRunner Model
....* Baseline model
....* Federated Learning Model

SummaRuNNer architecture


3. Evaluation (/programs/Evaluation)
..* Evaluator class
....* Convert idx to text
....* Extract gold and predicted summaries
....* Compute Rouge Scores
..* Compare Model Results





