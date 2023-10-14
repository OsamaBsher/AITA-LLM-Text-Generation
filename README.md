# AITA Comment Generation using LLMs

The project trained LLM models using datasets created from the AITA subreddit to generate comments. The following table of contents shows the different sections for the readme.

##### Table of Contents  
[Datasets](#datasets)  
[Requirments](#requirments)  
[Hyperparameter Optimization](#hyperparameter-optimization) 
[Training a model](#training)  
[Running a test with a trained model and a reddit post](#testing)  
[Solution](#solution)  
[Evaluation](#evaluation)  
[Overall Aim and Learning](#overall-aim-and-learning)

## Datasets

The stages for dataset prepration are as follows 

1. Dataset collection
    1. [Pushshift API](https://github.com/pushshift/api) was used to collected all post IDs on the AITA subreddit from its creation to April 1st 2023.
        - The logic for this can be found in ./data/pushshiftScraping   
        - The run was able to collect 1.6 Million post IDs
        - The post title, name and comments then needed to be fetched
    2. [Praw](https://praw.readthedocs.io/en/stable/)  was used to fetch the post title, body, verdict, comments
        - Praw limits the number of requests that can be sent per minute, to scrape 1.6 million posts it would have taken 33 days
        - So we created and ran 100 bots in parallel to scrape the 1.6 Million AITA subreddit posts, this took 8 hours
        - The logic can be found in ./data/prawScraping 
        - The data is separated into 100 csv files
    3. Data merging
        - To merge the 100 csv files, the files were merged into a single CSV file, this created a file that is 2.68GB in size
        - The logic can be found at ./data/mergeScrapedData
2. Data processing
    1. First some preprocessing was done to remove empty and deleted posts from the CSV files, and remove posts that do not contain the verdict
        - The logic can be found at ./data/dataPreProcessing
        - The processing was done in stage due to the large size of the dataset
    2. Then the data was cleaned
        - This included remove new lines characters, lower casing, removing posts without 2 comments atleast
        - posts with a length less than 10 tokens were removed
        - comments with less than 5 tokens were removed
        - posts that had the 2 top comments disagree on the verdict were removed
        - the post verdict was updated to match the verdict of the comments
        - the final csv created contained the following headers [id, title, text, verdict, comment1, comment2, score]
        - The logic for this can be found at ./data/dataCleaning
3. Data preparation
    1. The dataset was split into the different experiments to be tested, in total 8 splits were created
        - Splits (2 x 2 x 2 = 8) 
            - title
                - with 
                - without
            - token length
                - \>10 words <512 words  
                - \>10 words <1024 words 
            -  number of comments per posts
                - single 
                - double
        - The logic for this can be found ./data/dataSplitting
    2. After which data exploration was conducted, during this we discovered the proporition of not the asshole posts is much higher than asshole posts
        - The logic for the exploration can be found in ./notebooks/dataExploration.ipynb 
    3. To resolve this the datasets were balanced, the not the asshole posts with the highest upvotes were kept, equal to the number of asshole posts
        - The logic for this can be found in ./data/dataBalancing
    4. To ensure a fair test between the different models, the dataset was split into train, validation and test sets
        - The split was 0.8 : 0.1 : 0.1
        - This logic can be found in ./data/TrainTestValSplits

## Requirments

```sh
transformers=4.18.0
scikit-learn=0.24.1
torch=1.10.2
rouge_score=0.1.1
bert_score=0.3.13
evaluate=0.4.0
optuna=3.1.0
praw=7.6.1
requests=2.28.1
```

## Hyperparameter Optimization

We utilized Optuna for automated hyperparameter tuning, with the main script found at ./experiments/optuna.py. The model configuration to be optimized is specified in ./experiments/model_config.json. While Optuna was responsible for varying the learning rate, the selection of the model and dataset subset was determined by the configuration file.

Due to memory constraints, we could only experiment with the learning rate as our primary hyperparameter, as using a batch size larger than 2 resulted in out-of-memory errors. Our initial plan was to conduct 10 trials for both models on a single dataset and apply the best learning rate across all models. However, this proved to be impractical, as running 10 trials for a single model would take over 10 days of continuous computation on an a100 GPU.

As a result, we opted to train our models with a standard learning rate of 2e-5. In the future, given additional time and computational resources, determining the optimal learning rate would further enhance the model's performance.

## Training

Clone the [Bart Score repo](https://github.com/neulab/BARTScore) into the models folder to be used for training evaluation, and add its path to the line 

```sh
{
sys.path.insert(1, './BARTScore-main')
}
```

To train the model run the ./models/trainer.py, with the confugration of the model in ./models/model_config.json. The combination of all the models was 60GB in size, so they are not included here.

```sh
{
    "dataset": "balanced_subset_without_title_less_than_1024_single_comment", 

    "model": "bart", 
    
    "token_sequence_length": 1024,

    "learning_rate": 2e-5,
    
    "batch_size": 2,
    
    "num_train_epochs": 30
}
```

The models available are 
- T5
- Bart

The token seqeunce length should match the dataset
- 512 
- 1024 (Max input length for t5 is 512, cannot use 1024)

The epochs does not matter due to early stopping.

Input options 
1. Datasets
    - balanced_subset_with_title_less_than_512_double_comment 
    - balanced_subset_with_title_less_than_512_single_comment
    - balanced_subset_with_title_less_than_1024_double_comment 
    - balanced_subset_with_title_less_than_1024_single_comment 
    - balanced_subset_without_title_less_than_512_double_comment
    - balanced_subset_without_title_less_than_512_single_comment
    - balanced_subset_without_title_less_than_1024_double_comment 
    - balanced_subset_without_title_less_than_1024_single_comment
2. Model
    - "t5"
    - "bart"
3. Token sequence length (match dataset length, t5 only 512)
    - 512
    - 1024

## Testing

To test a model use the ./models/test_model.ipynb, include the path to the model by changing the model_path and include the url to the reddit post post_url. See example with best model.

```sh
post_url = "https://www.reddit.com/r/AmItheAsshole/comments/12szhlr/aita_for_pressuring_my_so_to_change_her_dress/"
model_path = "/Users/osama/Downloads/finalModels/bart/bart_balanced_subset_without_title_less_than_1024_double_comment"
```

Here is an example output, it contains the post title, post content and outputs comments using different search algorithms and different lengths.

```sh
Post Title: aita for pressuring my so to change her dress before a professional event?

Post Body: this morning my so was dressing up for a research symposium she's attending today. the one in which her senior class presented their research took place on monday, but this time around it is the underclassmen who are presenting their research. on monday, the seniors were instructed to dress in business casual, but today there was no explicit dress code for those just attending the event.      given this is still a professional academic event, i expected the norm to still be business casual. my so went with a dress that looked very much [like this](https://i.imgur.com/qtlvgd5.png) but a much brighter yellow, and sleeves that only went to mid-forearm.  i told her that maybe she shouldn't wear this dress and go with something business casual instead, to which she insisted that this was business casual. i replied saying it was casual but not business, and we went back and forth like that a little bit. ultimately it's her call what she wears and i don't typically try to change that, but this was a professional event, and i subjectively didn't think that dress looked appropriate the occasion.  i never was a big fan of this dress to begin with, and called it a grandma dress before, but the reason i had issue with it today wasn't out of personal distaste but for her professional image as these are the people she will be networking with for the rest of her career. at one point i said "i just don't want you to embarrass yourself" and her response seemed to indicate that she didn't believe that was my real reason, so i thought she suspected i was just saying this because i personally didn't like the dress.  after that back and forth, i thought about that last point and wanted to clarify that this wasn't about my personal taste. what i said came out something like this: "i want you to express yourself and it makes me happy when i see you being happy in how you dress, but i only say this because this is a professional event and i want people to take you seriously". for some context, many of her classmates  in her small cohort and even some faculty had often been dismissive of her inputs, so that was an insecurity she had shared with me before, so i really upset her when i said "i just want people to take you seriously" because she essentially heard "people don't take you seriously" as if i was trying to change her dress in an attempt to *fix* her image as if it was something needing fixing. i could have been more clear by saying "people might find that dress a little silly" or something else that made it clear i was only talking about the dress and not her personally, if that makes sense.  for more context, she is in healthcare, so she's worn scrubs to school almost every day, and she hasn't had many chances to express herself. in the end, she decided to go back and change into her scrubs, which i feel really guilty about.      i don't think she was ta at all, but is this a nah situation, or aita?

greed search comment, comment length - 0: yta. you don’t get to dictate what other people wear. it’s a professional event, not a school event.

search with 2 beams comment, comment length - 0: yta. you don’t get to dictate what she wears.

search with 10 beams comment, comment length - 0: yta. you don’t get to dictate what other people wear. it’s not your place to tell her what to wear.

greed search comment, comment length - 100: yta. you don’t get to dictate what other people wear. it’s a professional event, not a school event. you’re not entitled to dictate how other people dress, especially when it‘s not your place to tell her what to wear. you could’ve just said “i‘m not comfortable with it” and left it at that. instead, you went back and forth and made a big deal out of it.

search with 2 beams comment, comment length - 100: yta. you don’t get to dictate what she wears. she’s a senior, not a child. you’re not her parent. you sound like a bratty child.  you‘re not the only one who has a problem with this dress, and it‘s not your job to fix her image. you can‘t dictate what other people wear, and you can certainly not dictate what others wear. you could‘ve just said ‘i don‘ and left it at that.

search with 10 beams comment, comment length - 100: yta. you don’t get to dictate what other people wear. it’s not your place to tell her what to wear. you could’ve just said “i’m not comfortable with this dress” and left it at that. instead, you went back and forth and tried to make it seem like you’re trying to control what she wears. you sound like you have a lot of growing up to do, and you need to grow up.
```

## Solution

### Model Selection and Fine-tuning
For this project, we focused on fine-tuning two pre-trained models, T5 and BART, for the task of conditional text generation. Our goal was to train these models to generate comments reflecting moral judgment based on post and comment pairs from the r/aita subreddit. We utilized the T5 and BART models provided by the Hugging Face Transformers library. Both models were chosen due to their strong performance in text generation tasks.

### Configurability and Adaptability
By using a configuration file for model parameters, we have made our solution easily configurable, allowing other researchers to reconfigure the architecture and model parameters for their own purposes. This approach also aids our experimentation and evaluation process, enabling us to adapt the models to different tasks and datasets efficiently.

### Limitations and Future Work
We initially planned to include GPT-2 in our experiments, but due to time constraints, we were unable to train and test it within the deadline. In the future, we aim to explore the performance of GPT-2 on our task and compare it with the results obtained from the T5 and BART models.


## Evaluation

To compare the performance of the different models and on different data splits. For comment generation it is difficult to decide on a metric to compare the different models, therefore, we used a range of metrics, to identify a metric that can be used to compare the different approaches. The following metrics were calculated. We were inspired by metrics found in weeks 9-10 of the course.

1. Simple accuracy
    - We check if the generated comment sentiment matches the verdict given to the reddit post by the AITA subreddit 
    - This is a percentage
2. BLEU score 
3. Meteor score
4. Bert score
5. Bart score
6. Rogue score

Furthermore, we are planning to conduct a survey to ask people to rank comments blind, for a reddit post, some of the comments would be human generated and others by Bart and T5. This survey will run from code submission to the final report deadline, and will be used as an extra evaluation metric.

Furthermore, for comment generation using the model, there are multiple approaches to find the best comment, we plan on evaluating the greedy search, beam search and comment length paramaters using the survey aswell.

## Overall Aim and Learning

Our project aimed to generate comments with moral reasoning for posts with moral dilemmas using the AITA subreddit as a dataset. We learned about the application of transfer learning and fine-tuning for text-to-text generation and applied these methods to the field of morality. We collected a new dataset of posts and comments from AITA, including high-quality samples of a post and its top comments, together with the verdict. This dataset is state-of-the-art, as no dataset of this size exists for this subreddit.

During our project, we had the opportunity to work with state-of-the-art NLP technology and discovered that training transformer models can be both computationally expensive and time-consuming.  We were unable to train and test GPT-2, as originally planned, due to the time constraints. The training and testing of a single run for our models took up to 10 days, and we had eight datasets and two GPUs, which would have taken 40 days to complete. Despite this limitation, we were able to conduct research and generate morality-related comments using T5 and BART models, which provided us with valuable insights into natural language processing and its applications in the field of morality.

Through this project, we gained valuable experience in all aspects of NLP, including data collection, preprocessing, model selection, fine-tuning, and text generation. Our contribution of creating a new dataset, which includes high-quality samples of posts and comments from AITA, and generating morality-related comments using state-of-the-art transformers in an unexplored field, provides a strong foundation for future research in this area.




