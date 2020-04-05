## Word2Vec

Two architecture:

- CBOW (Cotinuous Bag-of-words):
  $$
  p(w_i \mid w_{i-h}, \dots, w_{i+h})
  $$

- Continuous Skip-gram:
  $$
  p(w_{i-h}, \dots, w_{i+h} \mid w_i)
  $$

Two ways to avoid softmax:

- Negative sampling
- Hierarchical softmax



## Paragraph2Vec aka Doc2Vec

Paragraph2vec goes from the paper. Doc2vec name goes from gensim library where it is implemented.
$$
\underbrace{\text{And the only reason for being a bee}}_\text{contexts} \space \underbrace{\text{bee}}_\text{focus word} \space \underbrace{\text{that I know of is making honey}}_\text{contexts}
$$
Remember that in word2vec, we had to two architectures

	- produce contexts given a focus word 
	- produce a focus word given some contexts

In doc2vec, we treat the document the same way as we treated words. There are two architectures. $d$ is a document id.

- DM (Distributed Memory):
  $$
  p(w_i \mid w_{i-h}, \dots, w_{i+h}, d)
  $$

- DBOW (Distributed Bag of Words):

  - similar to skip-gram model
  - Instead of the focus words, we condition on the document

  $$
  p(w_{i-h}, \dots, w_{i+h} \mid d)
  $$

We can use it to provide some documents similarities and apply for ranking.



## Sent2Vec

First ideas:

- Average pre-trained word vectors (word2vec, GloVe, etc)
- Maybe use TF-IDF weights for averaging
- Not so good idea because those pre-trained vectors are trained with some other objectives and they might not suit well for our task.

Sent2vec:

- Represent the sentences as a sum of sub-sentence units.

- Learn sentence embedding as a sum of sub-sentence units:
  $$
  sim(u, s) = \frac{1}{\mid G_s\mid} \sum_{g \in G_s} \langle \phi_u, \theta_g \rangle
  $$
  where $G_s$ is a set of word n-grams for the sentence s.

  