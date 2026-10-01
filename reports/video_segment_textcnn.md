# Video segment: TextCNN (about 90 seconds)

**Fill the {braces} from your own Colab run before you record. Say it in your own words; read this aloud once and then drop the script.** Speak about 150 words per minute, so this is roughly 1:30.

## Script

**[On screen: one slide, three small diagrams side by side: BiLSTM, TextCNN, DistilBERT, with the TextCNN highlighted]**

"Our study needed three neural models that really differ, not three versions of the same idea. The BiLSTM reads the whole tweet in order. DistilBERT lets every word attend to every other word. My model, the TextCNN, does something different again.

**[On screen: the TextCNN diagram: embedding, parallel filters of width 1 / 3 / 5, max-pool, concatenate, softmax]**

A TextCNN slides small filters over the tweet. A filter of width three is a detector for a pattern of three consecutive words, and max-pooling keeps only the strongest match anywhere in the tweet. So it sees local word patterns, but it has no memory of the rest of the sequence. That is a genuinely different assumption about what matters in the text.

**[On screen: the kernel comparison bar chart, `reports/figures/textcnn_kernel_comparison.png`]**

Why was that worth testing here? Our data analysis showed that a bag-of-words model with no word order already reaches about 99.7% accuracy. So the real question is how much word order adds. The kernel width lets us test that directly: width one is just a single-word detector, and wider kernels add local order. In our validation runs, {one sentence: what happened, for example 'wider kernels did not beat width one by more than the seed-to-seed spread'}.

**[On screen: the stress-test table]**

And when we masked the strongest class keywords, {one sentence: how far the TextCNN and TF-IDF fell}. When we shuffled word order, {one sentence: what changed}.

What the TextCNN cannot do is remember anything beyond its window, so any pattern that depends on the whole tweet is out of its reach. It trains quickly, about {X} seconds per run on a {GPU}, which made running three seeds per configuration practical, and that is what let us say which differences are real."

## Timing guide
- Why a third, different architecture: 0:00 to 0:25
- How it works: 0:25 to 0:50
- Why kernel width tests our research question, and the result: 0:50 to 1:15
- Stress tests and limitation: 1:15 to 1:30

## What you must not say
- Do not say the TextCNN "captures sequential dependencies" without qualification. Its filters only see a fixed window, so its sequential modelling is local. Overclaiming here is exactly what a grader will probe.
- Do not call a model "better" unless the gap is larger than the seed spread.
- Do not quote dry-run numbers. Use only your own Colab results.

## Questions you should be able to answer (individual defence)
1. Why global max-pooling and not average-pooling? (It keeps the strongest local match anywhere in the tweet, so position does not matter; average-pooling dilutes a rare strong signal across 70 positions.)
2. What does a filter of width 1 learn? (A weight vector over one word's embedding, so in effect a learned, max-pooled word detector.)
3. Why is the embedding layer the biggest part of the model? (20,000 x 128 = 2.56 million of about 2.6 million parameters, so the filters are a small part.)
4. Why did you use class weights, and what did the unweighted run show?
5. Why three seeds? What would a single seed have hidden?
6. Why is the clean test subset the headline and not the full test set?
7. Keyword masking collapsed the model. What does that say about the task, and about real use for routing support requests?
8. What does your model do that TF-IDF + logistic regression does not? (Be honest: answer from your own stress-test and error-analysis results.)
9. Why is DistilBERT expected to behave differently, and did it?
10. Which decision in your tuning would you change if you had another week?
