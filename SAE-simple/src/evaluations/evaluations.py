from transformers import pipeline
import numpy as np
import json
import tqdm


def sentiment_classify(prompts, no_steers, steers, sentiment_file=None):
    # Initialize sentiment classifier
    classifier = pipeline('sentiment-analysis', model='/home/ckqsudo/code2024/0models/sentiment-roberta-large-english', device=0)
    print("Writing outputs to ", str(sentiment_file))

    # accuracies = []

    # Loop through each prompt and corresponding generations
    pos_no_steer,pos_steer=[],[]
    
    for i in tqdm.tqdm(range(len(prompts)), desc='Scoring generation sentiments'):
        prompt = prompts[i]
        no_steer = no_steers[i]
        steer = steers[i]

        sentences_for_prompt = []
        sentences_for_prompt.append(f'{no_steer}')
        sentences_for_prompt.append(f'{steer}')
        try:
            predictions_for_prompt = classifier(sentences_for_prompt, max_length=512)
            assert len(predictions_for_prompt)==2
        except IndexError:  # Handle cases where the generation is too long
            print("Exception occurred, please check")
            predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)

        # Calculate the proportion of "positive" sentiment
        print(predictions_for_prompt)
        
        pos_no_steer.append(float(predictions_for_prompt[0]["label"] == "POSITIVE"))
        pos_steer.append(float(predictions_for_prompt[1]["label"] == "POSITIVE"))
        print(pos_no_steer,len(pos_no_steer))
        
        # positive_proportion = positive_proportion / len(predictions_for_prompt)
        # accuracies.append(positive_proportion)

        # Save the sentiment predictions to a file if a sentiment file is provided
        # if sentiment_file is not None:
        #     with open(sentiment_file, 'a') as fo:
        #         for res in predictions_for_prompt:
        #             fo.write(json.dumps(res) + '\n')
    print("no steer",float(sum(pos_no_steer))/len(pos_no_steer))
    print("steer",float(sum(pos_steer))/len(pos_steer))
    # print(accuracies)
    # return np.nanmean(accuracies), np.std(accuracies)
    


# Example usage:
# Assuming you have the lists prompts, no_steers, and steers populated
with open("/home/ckqsudo/code2024/CKQ_ACL2024/Control_Infer/SAE-simple/src/evaluations/res.json", "r", encoding="utf-8") as gen_f:
    gen = json.load(gen_f)

prompts = []
no_steers = []
steers = []

for i in range(1, len(gen)):
    prompts.append(gen[i]["prompt"])
    no_steers.append(gen[i]["no_steer"][1])  # Assuming the second item is the actual text
    steers.append(gen[i]["steer"][1])  # Assuming the second item is the actual text

# Call sentiment_classify function with the generated lists
sentiment_classify(prompts, no_steers, steers, sentiment_file="./score.json")
