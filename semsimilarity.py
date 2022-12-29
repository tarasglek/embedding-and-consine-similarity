from sentence_transformers import SentenceTransformer, util
import torch
import hashlib
model = SentenceTransformer('all-MiniLM-L6-v2')


# Two lists of sentences
sentences1 = ['The cat sits outside',
             'A man is playing guitar',
             'The new movie is awesome',
             'in open air Feline sat',
              'A woman watches TV',
              'The new movie is so great',
              'feline is dealing drugs under bridge',
              """Kat loves to sit in ze garden und watch ze birds. Her fur iss beautiful und shiny, und she always has a satisfied look. She iss a very laid-back Kat und seems to never get stressed.

One day, ze Kat decided to take a nap. She lay down on her favorite spot in ze grass und closed her eyes. It was a beautiful day, ze sun was shining und a light breeze was blowing. Ze Kat was so relaxed that she soon fell asleep.

When she woke up, she felt refreshed und ready to enjoy ze day. She stood up, stretched und headed into ze house. She knew her owner missed her und she looked forward to providing him with company."""]

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
# embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings1[0])

indexes = sorted(map(lambda x: x[0], enumerate(cosine_scores)), key=lambda x: cosine_scores[x][0], reverse=True)

def trim_string(string, max_length=50):
    trimmed = string[:max_length]
    if len(string) > max_length:
        trimmed += "..."
    return trimmed

#Output the pairs with their score
for ii in range(len(indexes)):
    i = indexes[ii]
    right = 0
    # print(f"{sentences1[i]}", hashlib.sha256(embeddings1[i].cpu().numpy()).hexdigest(), len(embeddings1[i]))
    print(f"'{trim_string(sentences1[i])}' vs '{trim_string(sentences1[right])}' Score:{cosine_scores[i][right]:.4f}")

