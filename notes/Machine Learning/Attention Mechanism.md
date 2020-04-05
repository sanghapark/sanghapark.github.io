# How does Attention work?

### Step 1: Compute a score each encoder state

1. encode the input sequence into set of internal states (h1, h2, h3, h4, h5). 
2. Using two components (all the encoder states and the current state of the decoder), we will train feed forward nn. Why?
3. We are trying to predict the first word in the target sequence. We do not need all the encoder states to predict the first word. But we need those encoder states which store infor about the word "Rahul" in the input sequence. Suppose the info is presend in the states, h1 and h2.
4. We want our decoder to pay more attention to the states h1 and h2 while paying less attention to other states of the encoder.
5. We train feed forward nn which will learn to identify relevant encoder states by generating a high score for the states for which attention is to be paid while low score for the states which are to be ignored.
6. Let s1, …, s5 be the score for the states h1, …, h5. We want s1 and s2 are high while other are relatively low.

### Step 2: Compute the attention weights

1. Once these scores are generated, we apply softmax on these scores to produce the attention weights e1,…, e5. All the weights like between 0 and 1. All the weights sum to 1.
2. e1=0.75, e2=0.2, e3=0.02, e4=0.02, e5=0.01 means the decoder needs to put more attention on the states h1 and h2.

### Step 3: Compute the context vector

1. Once we have computed the attention weights, we need to compute the context vector (thought vector) which will be used by the decoder in order to predict the next word in the sequence.

   context_vector = e1h1 + e2h2 + … + e5h5

### Step 4: Concatenate context vector with output of previous time step

1. The decoder uses the two input vector s to generate the next word in the sequence
   - The context vector
   - The output word generated from the previous time step

2. Simpy concatenate these two vectors and feed the merged vector to the decoder. For the first time, since there is not output from the previous time step, we use a special <START> token for this purpose

### Step 5: Decoder Output

1. The decoder then generates the next word in the sequence and along with the output, the decoder will also generate an internal hidden state, d1.

![1*9Djuu6M-jPANRYg_tWFqUQ](https://miro.medium.com/max/2000/1*K9NXW3w7O_1p5sRm6NnEuQ.jpeg)

![1*9Djuu6M-jPANRYg_tWFqUQ](https://miro.medium.com/max/2000/1*9Djuu6M-jPANRYg_tWFqUQ.jpeg)



![1*tfsaoVIVNr3dk-PaIpn6oQ](https://miro.medium.com/max/2000/1*tfsaoVIVNr3dk-PaIpn6oQ.jpeg)

![1*Xh1qGsjfhjox4v3UZ4iqRw](https://miro.medium.com/max/2000/1*Xh1qGsjfhjox4v3UZ4iqRw.jpeg)

![1*DpgRp8JWQnaVZM8b9Y_0Bg](https://miro.medium.com/max/2000/1*DpgRp8JWQnaVZM8b9Y_0Bg.jpeg)

Unlike the fixed context vector used for all the decoder tiem steps in case of the traditional S2S models, we compute a separate context vector for each time time step by computing the attention weights every time.



Thus using this mechanism our model is able to find interesting mappings between different parts of the input sequence and corresponding parts of the output sequence.



























