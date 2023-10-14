# importing pandas
import pandas as pd
  
combine_array = []

for x in ["subreddit_posts_text_{}.csv".format(index + 1) for index in range(100)]:
    combine_array.append(pd.read_csv(x))

result = pd.concat(combine_array)

result.to_csv("full_dataset.csv", encoding='utf-8', index=False)