from sentence_transformers import SentenceTransformer, util
import torch
import hashlib
model = SentenceTransformer('all-MiniLM-L6-v2')


# Two lists of sentences
sentences1 = ['The cat sits outside',
             'A man is playing guitar',
             'The new movie is awesome',
             'Feline seated in open air',
              'A woman watches TV',
              'The new movie is so great',
              'feline is dealing drugs under bridge']

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
# embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings1[0])

indexes = sorted(map(lambda x: x[0], enumerate(cosine_scores)), key=lambda x: cosine_scores[x][0], reverse=True)

print(indexes)
#Output the pairs with their score
for ii in range(len(indexes)):
    i = indexes[ii]
    right = 0
    # print(f"{sentences1[i]}", hashlib.sha256(embeddings1[i].cpu().numpy()).hexdigest(), len(embeddings1[i]))
    print(f"'{sentences1[i]}' vs '{sentences1[right]}' Score:{cosine_scores[i][right]:.4f}")

