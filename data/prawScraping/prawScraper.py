import praw
import pandas as pd
import csv
import json
from multiprocessing import Process
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

# Read in the list of post ids
df = pd.read_csv("posts_ids_and_scores.csv")
use_posts = df['id'].tolist()

# Read my keys
with open("keys.json") as f:
    keys = json.load(f)

chunks = np.array_split(use_posts, len(keys))
arguments = []

for index, chunk in enumerate(chunks):
    arguments.append([keys[index], chunk, "subreddit_posts_text_{}.csv".format(index + 1)])

def fetch_data(data):

    keys, ids, csv_file = data

    # Initialize reddit API
    reddit = praw.Reddit(client_id=keys['client_id'],
                        client_secret=keys['client_secret'],
                        user_agent=keys['user_agent'])

    
    f = open(csv_file,"w",encoding="utf-8") 
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["id", "title", "text", "edited", "verdict", "comment1",  "comment2",  "comment3",  "comment4",  "comment5",  "comment6",  "comment7",  "comment8",  "comment9",  "comment10", "score", "url", "time_created"])

    verdicts = ["YTA", "NTA", 'ESH', 'NAH']

    counter = 0

    for idx in ids:


        if counter % 1 == 0:
            print(counter, "/", len(ids))
        counter += 1

        post = reddit.submission(idx)
        verdict = post.link_flair_text

        if not verdict:
            verdict =  "NA" 

        post.comment_sort = "top"
        comments = post.comments.list()
        total_comments = []

        if(len(comments) <= 2): # if post has less than 10 comments skip
            continue

        for top_level_comment in comments: # extract comments
            try:
                if not top_level_comment.body:
                    continue
            except:
                continue
            if(len(total_comments) == 10):
                break
            if("*I am a bot, and this action was performed automatically." in top_level_comment.body): # ignore bot comments
                continue
            if(not any(label in top_level_comment.body for label in verdicts)): # if comment does not have label ignore
                continue
            total_comments.append(top_level_comment.body)

        if len(total_comments) < 3:
            continue

        if len(total_comments) < 10:
            for i in range(10 - len(total_comments)):
                total_comments.append("")

        line_stuff = [idx, post.title, post.selftext, str(post.edited), post.link_flair_text, 
                        total_comments[0], 
                        total_comments[1], 
                        total_comments[2], 
                        total_comments[3], 
                        total_comments[4], 
                        total_comments[5], 
                        total_comments[6], 
                        total_comments[7], 
                        total_comments[8], 
                        total_comments[9], 
                        post.score, post.url, post.created_utc]
        writer.writerow(line_stuff)

    f.close()


def run_in_parallel():
    args = arguments
    pool = Pool(processes=len(args))
    pool.map(fetch_data, args)

if __name__ == '__main__':    
    run_in_parallel()
