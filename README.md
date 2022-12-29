Can combined suggestions in
https://medium.com/@nils_reimers/openai-gpt-3-text-embeddings-really-a-new-state-of-the-art-in-dense-text-embeddings-6571fe3ec9d9

with https://www.youtube.com/watch?v=coaaSxys5so

Sample:
```
time python semsimilarity.py
'The cat sits outside' vs 'The cat sits outside' Score:1.0000
'in open air Feline sat' vs 'The cat sits outside' Score:0.3735
'Kat loves to sit in ze garden und watch ze birds. ...' vs 'The cat sits outside' Score:0.3242
'feline is dealing drugs under bridge' vs 'The cat sits outside' Score:0.2835
'A woman watches TV' vs 'The cat sits outside' Score:0.1310
'A man is playing guitar' vs 'The cat sits outside' Score:0.0363
'The new movie is so great' vs 'The cat sits outside' Score:-0.0029
'The new movie is awesome' vs 'The cat sits outside' Score:-0.0247
```