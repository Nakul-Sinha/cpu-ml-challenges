# Approach: ParseRift (predicting parser attachment disagreement)

## Summary
For every token in a sentence the task predicts a single bit: would a population of 25 independent
dependency parsers disagree on which word it attaches to. The label is a property of the whole
sentence context, not of the token alone, so the model has to read the surrounding words. My
solution is a character aware BiLSTM tagger trained from scratch on CPU, with the decision threshold
calibrated for MCC.

## Token representation
Each token is encoded by 3 complementary parts:
1. A learned word embedding, with a shared unknown vector for tokens seen fewer than 2 times in
   training. About 45 percent of test tokens are unseen, so word identity alone is not enough.
2. A character CNN over the token characters (parallel convolutions of width 3, 4, and 5 with max
   pooling). This gives every token, including unseen ones, a representation from its spelling and
   captures morphology and punctuation shapes.
3. A small set of shape features: leading capital, all caps, all lowercase, contains a digit, is
   pure punctuation, single character, length, is alphabetic, contains an apostrophe, is a common
   punctuation mark.

## Sequence model
The per token representations feed a 2 layer bidirectional LSTM that reads the whole sentence, so
each token prediction is conditioned on its left and right context. This is the productive signal:
a preposition, a coordinator, or a word that could be an auxiliary or a main verb is contested
because of what surrounds it, not because of the word itself. A small MLP head maps each token state
to a logit.

## Training and calibration
Loss is binary cross entropy with a positive class weight to counter the 23 percent contested rate.
I train an ensemble of 3 seeds and average their probabilities, which reduces variance. The decision
threshold is not left at 0.5: it is swept on a held out sentence split to maximize MCC directly,
which matters because MCC punishes both a trivial constant predictor and a low precision high recall
rule. The ensemble is then retrained on all training data for the final prediction, using the
calibrated threshold. The split is by sentence so no sentence leaks between fit and calibration.

## Local validation
Sentence level held out MCC is about 0.34 (a single model reaches about 0.34 by epoch 16; the seed
ensemble is at least as high). This is above the char aware BiLSTM reference (0.258) and above the
0.32 target. The whole pipeline trains in a few minutes on CPU.

## Compliance
Trained from scratch on the provided tokens and labels only. No external parsers, taggers,
treebanks, pretrained embeddings, or internet. token_id, sentence_id, and position carry no signal
and are used only to group and order tokens; the prediction comes from the words and their context.

## Time spent
About TBD.
