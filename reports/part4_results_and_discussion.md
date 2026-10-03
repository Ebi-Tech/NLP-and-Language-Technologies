# Part 4: Results, Discussion and Error Analysis

This section covers the DistilBERT model, the comparison of all five models on the test split, and
a look at where the models get things wrong. Every number here comes from a file in
`reports/results/`, and the file is named next to the number. Part 3 explains the setup.

## 5. Results

### 5.1 DistilBERT: how the text was turned into model input

DistilBERT [12] is a smaller version of BERT [13]. It keeps the self attention that lets every word
look at every other word, but with fewer layers. It was picked for two reasons that come straight
out of Part 1.

The first is subword tokenization. Part 1 found that half of the words in this dataset (50.7%)
show up exactly once. The BiLSTM and TextCNN map all of those to a single unknown word token, so
they lose them. DistilBERT splits them into smaller pieces it already knows, so a rare or
misspelled word still means something.

The second is pretraining. The other two neural models learn their word representations from the
27,755 training tweets and nothing else. Only 188 and 217 tweets in the whole dataset belong to the
two smallest classes, which is very little to learn from. DistilBERT starts with language knowledge
it already has.

This model uses the raw `tweet` column with only URLs and @handles taken out, not the shared
`tweet_clean` column. The reason is that the tokenizer was pretrained on normal text, so it expects
text that still looks normal. Taking out punctuation or case would throw away things the pretrained
weights already know how to use. `distilbert-base-uncased` lowercases the text itself, so this is
still fair against the other models on that point.

The sequence length was measured instead of copied from Part 1. On a sample of 3,000 tweets the
token counts are 56 at the median, 72 at the 90th percentile, 84 at the 99th and 105 at the most.
Part 1 reported a maximum of 68 whole words, so splitting words into pieces makes the sequences
about half again as long. The 70 step limit used by the BiLSTM and TextCNN would have cut off more
than 10% of tweets here. A limit of 128 tokens fits every tweet in the sample with nothing cut off.

Training used AdamW [7] with a learning rate of 2e-5, batch size 32, three epochs and PyTorch seed
42. It took 15.4 minutes on a Tesla T4 GPU (`distilbert_run_info.json`).

### 5.2 The class weight experiment

The one tuning decision for this model was whether to use `balanced` class weights in the loss,
which is the choice Part 2 made for the BiLSTM. Both versions were trained with everything else
kept the same and compared on validation only.

| Configuration | Validation macro F1, epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| With class weights | 0.9844 | 0.9953 | 0.9950 |
| Without class weights | 0.9999 | 0.9999 | 0.9997 |

Both runs are logged in `experiments_distilbert.csv` with the hypothesis and the decision.

The version without class weights was kept, but two things need saying about that.

The gap at the last epoch is 0.0047. Part 3 set a rule before any training started: anything inside
0.005 counts as a tie, and the tie is broken by picking the model with fewer parameters. These two
have the same number of parameters, so the rule does not actually pick a winner. The honest way to
report this is that the two configurations tied and the simpler loss was kept, not that class
weights hurt this model.

The other thing is that validation is used up. The unweighted model hits 0.9999 after one epoch and
its validation report rounds to 1.00 on precision, recall and F1 for all five classes. There is no
room left to tell configurations apart at this level. Both runs also used one seed each, so neither
number has a spread to compare against. Learning curves for both are in
`reports/figures/distilbert_learning_curve.png`.

### 5.3 All five models compared

The test split was scored once, after every configuration decision was already made. All five
models were then rescored by the same function from their saved predictions, so no model is scored
by different code (`comparison.ipynb`, which writes `model_comparison_all.csv`). The clean subset
is the 5,831 test tweets whose cleaned text does not appear anywhere in training, defined in
`src/leakage.py`.

| Model | Accuracy | Macro F1 | Clean macro F1 | Errors | Macro AUC |
|---|---|---|---|---|---|
| TextCNN | 0.9995 | 0.9978 | 0.9977 | 3 | 0.99999 |
| DistilBERT | 0.9993 | 0.9954 | 0.9952 | 4 | 0.99985 |
| TF-IDF and logistic regression | 0.9970 | 0.9813 | 0.9808 | 18 | 0.99997 |
| BiLSTM | 0.9963 | 0.9792 | 0.9787 | 22 | 0.99975 |
| Majority class | 0.8235 | 0.1806 | 0.1822 | 1,050 | 0.5 |

Accuracy and macro F1 come from `model_comparison_all.csv` and macro AUC from `roc_auc_all.csv`.
The clean columns come from each model's `_clean_test.json` file where it exists
(`distilbert_clean_test.json`, `textcnn_clean_test.json`, `tfidf_logreg_clean_test.json`) and are
recomputed from the prediction files otherwise. The BiLSTM row is computed from
`preds_bilstm_test.csv`. Part 2 reports a macro F1 of 0.9723 for the same model from a different
run, and that still has to be sorted out before this report is handed in.

Three things come out of this table.

Accuracy cannot tell the four trained models apart. They sit between 0.9963 and 0.9995. Macro AUC
cannot either, since all four are above 0.9997. That is why `reports/figures/comparison_roc.png`
has a zoomed panel next to the full one, because at full scale the curves sit on top of each other.

Macro F1 does tell them apart, but only just, and the gap rests on very few tweets. The difference
between first and second place is three errors against four.

The majority class baseline gets 82.35% accuracy with a macro F1 of 0.1806. That is the whole
argument from Part 1 in one line, and it is why macro F1 is the headline metric here.

The full test and clean test columns differ by less than 0.001 for every model, and the reason is
sharper than just the overlap being small. Every error from all four trained models falls inside
the clean subset. In other words, all 117 test tweets that have an exact copy in training are
classified correctly by all four models. Dropping them drops only correct answers, which is why the
clean score comes out slightly lower and not higher. The duplication Part 1 found is real, but on
this dataset it only pads accuracy with answers the models were getting right anyway. It does not
explain why any model looks strong.

### 5.4 Per class results

DistilBERT gets a per class F1 of 1.000 on `sexual_violence`, 0.998 on `physical_violence`, 1.000
on `harmful_traditional_practice`, 0.995 on `emotional_violence` and 0.984 on `economic_violence`
(`distilbert_test.json`).

The lowest score is on `economic_violence`, and it comes from one wrong tweet out of 32. Recall is
0.969 and precision is 1.000. A per class F1 built on 32 examples moves by about 0.03 for every
single tweet, so the order of the rare classes inside any one model is not a stable number and
should not be read as one.

All four confusion matrices are in `reports/figures/comparison_confusion_matrices.png`, and the
DistilBERT one on its own is in `reports/figures/confusion_matrix_distilbert.png`.

### 5.5 Stress tests

A score above 0.99 does not prove a model has learned to read these categories. Part 1 showed the
classes can mostly be told apart by a small set of words. So the test text was changed in three
ways, all using `src/robustness.py` on the full test set.

Masking swaps the ten strongest words for each class, picked on the training split only, for a made
up placeholder word. Random masking swaps the same number of randomly chosen words in the same
tweets. That one is the control, and it is the important one: it shows how much of the drop is just
from damaging the text rather than from losing those particular words. Shuffling scrambles the word
order inside each tweet with seed 42. Masking changes 99.9% of test tweets and swaps 2.2 words on
average.

| Model | Unchanged | Keyword masked | Random masked | Shuffled |
|---|---|---|---|---|
| TextCNN | 0.9978 | 0.3999 | 0.8914 | 0.9978 |
| DistilBERT | 0.9954 | 0.2463 | 0.9207 | 0.9857 |
| TF-IDF and logistic regression | 0.9813 | 0.2485 | 0.9351 | 0.9777 |
| BiLSTM | 0.9792 | not tested | not tested | not tested |

These are macro F1 scores from `stress_test_comparison.csv`. The empty cells are tests that have
not been run, not tests that showed no effect.

Masking takes away most of what every tested model had. DistilBERT drops from 0.9954 to 0.2463,
which is a fall of 0.7491. Its accuracy under masking is 0.8294, and a model that always answers
`sexual_violence` gets 0.8235. So the fine-tuned transformer lands within 0.006 accuracy of
guessing the biggest class every time.

The per class numbers in `distilbert_masked_test.json` show this is a collapse onto the majority
class and not a general drop in quality. Recall falls to 0.000 for both `economic_violence` and
`emotional_violence`. All 98 masked `emotional_violence` tweets get called `sexual_violence`. For
`physical_violence`, recall goes from 1.000 down to 0.072.

The control is what makes this mean something. Masking the same number of random words only costs
0.0747 of macro F1, against 0.7491 for the keywords. So about 90% of the collapse comes from those
roughly fifty specific words and not from the text being damaged. The TF-IDF baseline behaves
almost the same way, falling to 0.2485 masked and 0.9351 on the control. A pretrained transformer
and a bag of words model lean on the same vocabulary to about the same degree.

The TextCNN sits in the same place, just not quite as far. It falls to 0.3999 masked against 0.8914
on the control, so about 82% of its drop comes from the keywords. The same sum gives 90% for
DistilBERT and 94% for the TF-IDF baseline. All three lean on the same small vocabulary, and the
TextCNN leans on it a little less than the other two.

Shuffling only means something for a model that can use word order in the first place. The TextCNN
that was selected is one width 1 convolution with global max pooling, which ignores position by
design, so its identical shuffled and unshuffled scores say something about the architecture and
nothing about the data. That is decision 25 in the decision log. For DistilBERT the test is real,
and scrambling every word costs 0.0097, from 0.9954 down to 0.9857.

## 6. Discussion

### 6.1 The models barely use the sequence

The question behind this project is how well sequential models handle this problem. On this dataset
they do handle it, but not by acting sequential.

Two results point the same way. The best model overall is the TextCNN in a setup that cannot see
word order at all, since one width 1 kernel with global max pooling is really just a learned single
word detector. And DistilBERT with the word order of every test tweet destroyed still scores
0.9857, which beats the BiLSTM at 0.9792 and the TF-IDF baseline at 0.9813 with their text left
alone. A transformer reading scrambled text does better than the model that was picked for this
study specifically because it reads sequences in order. The 0.0097 that shuffling does cost
DistilBERT is the only number in this study that isolates what word order is worth here, and it is
small.

This matches research on other tasks. Pham et al. found models keep most of their accuracy on most
GLUE tasks when the words are shuffled at test time [16], and Sinha et al. found that pretraining
on word order randomised text costs surprisingly little later on [17]. What is reported here is a
narrower version of the same thing. These five classes are set apart by which words show up, and
the order they show up in adds very little that any of these models picks up.

Part 2 said the point of including a recurrent model was to test whether word order helps here, not
to assume it does. Read against the numbers above, that test came back negative. That is a finding,
not a failed experiment.

### 6.2 The high scores come from a separable vocabulary

The stress tests measure directly what the headline scores are built on. Taking out about fifty
words drops DistilBERT to roughly the accuracy of a model that always answers the same class, while
taking out the same number of other words costs 0.0747. So the dependence is on those words
specifically and not on the text being chewed up. The organizers frame this challenge as
classification without keywords, and by this measurement none of the tested models does that.

The more interesting part is that this is just as true for the pretrained transformer as for the
bag of words baseline. Pretraining should give it representations that let it generalise past
surface vocabulary, and the fine-tuned model does not use them that way. It has no reason to. When
a feature is almost perfectly predictive, a model will learn it instead of something harder to
pick up. This is what is called shortcut learning [18], and it has been documented in natural
language inference, where models were found to use annotation artifacts and shallow rules that
happen to line up with the label [19], [20], and in argument reasoning, where reported scores were
traced back to spurious cues in the data [21]. This dataset has an unusually strong cue of the same
kind, which Part 1 spotted in the baseline's feature weights before any neural model was trained.

So the ranking in Section 5.3 should not be read as a ranking of language understanding. The four
trained models are separated by between 3 and 22 errors out of 5,948, while all of them lose most
of their performance to the same change in the text. On this dataset the stress tests separate the
models far more than the test scores do.

### 6.3 Class weights show up in the errors

The two models trained with balanced class weights are the TF-IDF baseline and the BiLSTM. They
make 18 and 22 errors, and 14 and 20 of those are tweets whose real class is `sexual_violence`
being called something else. The two models trained without class weights, the TextCNN and
DistilBERT, make 3 and 4 errors, and only 1 and 2 of those go that way.

The reason is straightforward. Weighting the rare classes up makes a model more willing to move off
the majority class, and on a test set that is 82.3% majority class, most of those moves are going
to be wrong. This is the same thing behind the Stage C result in Part 3, where dropping class
weights raised the TextCNN validation macro F1 from 0.9831 to 0.9951. Here it shows up in the
actual errors of the final models and not only in a validation number.

This also limits what the comparison can claim. The BiLSTM differs from the TextCNN and DistilBERT
in its loss weighting as well as its architecture, so the gap between it and the other two neural
models cannot be put down to architecture on its own. An unweighted BiLSTM run would fix this. It
has not been done.

### 6.4 What the comparison does and does not show

Pretraining and subword tokenization did not win here. DistilBERT comes second to a much smaller
model. It has about 66 million parameters against roughly 2.6 million for the selected TextCNN,
nearly all of which sit in its embedding layer, and it needed 15.4 minutes of GPU training against
21 to 25 seconds per TextCNN run (`experiments_textcnn.csv`).

That is not because the transformer is weaker. It is because the task gives it nothing to be
stronger at. The ceiling here is set by how consistently the tweets are labelled, and that ceiling
is reached long before model size starts to matter. The one place the architectures do separate is
under the stress tests, where DistilBERT holds 0.9857 with the word order gone, and that is exactly
the thing the headline metric does not measure.

## 7. Error Analysis

### 7.1 All four DistilBERT errors

DistilBERT gets 4 of 5,948 test tweets wrong, so each one can be looked at directly instead of
summarised. The predictions are in `preds_distilbert_test.csv`.

**ID_2F20CBVV.** Labelled `emotional_violence`, predicted `sexual_violence` at 0.981. The tweet is
about being verbally abused on a bus, sworn at and called names, and it ends with a threat of rape.
Both the labelled class and the predicted class are in the text. The label only records one of
them.

**ID_1RXM1NG7.** Labelled `sexual_violence`, predicted `physical_violence` at 0.997. In one
sentence it says a husband beats his wife, forces sex on her, and raped their daughter. Three
categories at once. This is the only test tweet that all four trained models get wrong, and each
one puts it in a class the text also describes.

**ID_2PSQRJXD.** Labelled `economic_violence`, predicted `physical_violence` at 0.999. The tweet is
about a woman whose husband was fired from his job over his private life. The words for the right
class are right there, since both `fired` and `job` are among the ten strongest `economic_violence`
features, and the model still puts almost all its probability somewhere else. This is the one error
of the four that is not explained by several categories being in the text.

**ID_A6MZJ758.** Labelled `sexual_violence`, predicted `physical_violence` at 0.997. It describes
consensual spanking between adults, and says so clearly. Whether this tweet shows gender based
violence at all is doubtful, and the prediction just follows the physical words in it.

So three of the four errors are tweets that describe more than one category, and a fourth looks
mislabelled. What limits this model is not its ability to represent the text. It is a one label per
tweet scheme applied to posts that often report several kinds of violence together. The same thing
explains the three TextCNN errors, two of which are the same tweets.

One more thing about these four errors matters for any real use. The model gives the wrong class
between 0.981 and 0.999, and never gives the right class more than 0.018. It is not unsure where it
is wrong. That means no confidence cutoff would send these cases to a human while leaving the
correct answers alone. Calibration was not tested in this study.

### 7.2 Errors across the models

Across the four trained models, 39 different test tweets are wrong for at least one model, and only
one is wrong for all four. So the error sets barely overlap. The models fail on different tweets
even though they score within 0.019 macro F1 of each other. The biggest overlap is 4 shared errors
between the TF-IDF baseline and the BiLSTM, which are the two models trained with class weights.

Two confusions account for most errors in every model. `sexual_violence` called
`emotional_violence` is the single biggest cell for both the TF-IDF baseline at 7 tweets and the
BiLSTM at 12, and `sexual_violence` called `physical_violence` is next. Both pairs are classes that
share vocabulary and turn up together in the same post, which is the same cause as in the
individual errors above.

Part 1 flagged that `physical_violence` tweets are much shorter, 23.3 words on average against 34
to 42 for the other classes, and said to check whether models were keying on length instead of
content. The errors give no sign of that. `physical_violence` F1 is among the highest for every
model, 0.998 for DistilBERT, and the errors involving that class go in both directions rather than
piling up in one.

### 7.3 How the models fail when the keywords are gone

The masked condition fails in a different way from the normal one, and it is worth reading as error
analysis on its own. With masking, DistilBERT puts 825 of 892 `physical_violence` tweets and all 98
`emotional_violence` tweets into `sexual_violence` (`distilbert_masked_test.json`).

Once its strongest cues are gone, the model falls back on how common each class was in training
rather than on whatever weaker evidence is left in the tweet. And there is plenty left. Only 2.2
words per tweet were swapped out, against a median tweet length of 43 words. Most of what is needed
to classify these tweets is still sitting there and the model does not use it. That is a more
specific claim than just saying it depends on keywords.

## 8. Limitations

Everything here rests on one fixed split with the test set scored once. The two smallest classes
have 28 and 32 test tweets, so one wrong answer moves a per class F1 by about 0.03 and macro F1 by
about 0.006. First and second place are separated by three errors against four, which is well
inside that. No confidence intervals or significance tests were run. The order of TextCNN and
DistilBERT should not be treated as settled.

Seeds are uneven across the project. The TextCNN configurations were each run with three seeds and
are reported with standard deviations. DistilBERT is one seed and the BiLSTM sets no seed at all,
so neither can be compared against a seed spread. For DistilBERT the class weight decision rests
on one run each with a gap smaller than the project's own tie rule.

This part did not experiment much. One hyperparameter decision was tested for DistilBERT. The
learning rate, the number of epochs, the sequence length and the choice of pretrained model were
all left at normal values and never varied. There was no early stopping and no picking of the best
epoch, so the reported model is just whatever it looked like after three epochs. Pretrained word
embeddings were not tested for any model, which means the comparison between the pretrained
transformer and the two models trained from scratch mixes up pretraining with architecture.

The stress tests show keyword dependence but do not fully describe it. They use one placeholder
word, one way of picking keywords with ten per class, one random masking draw and one shuffle seed.
None of the three has been run for the BiLSTM, so that whole row in Section 5.5 is still empty.

The labels themselves are a limit, and the error analysis ran straight into it. The scheme gives one
category per tweet to posts that often describe several, and at least one test tweet looks like it
does not describe violence at all. There is no inter annotator agreement figure for this dataset,
and `Test.csv` has no labels, so there is no outside set to check whether these scores hold up.
Validation is also used up for the selected DistilBERT setup, which means tuning on validation has
no room left to tell anything apart at this level.

Last, the BiLSTM numbers used in the comparison come from the saved prediction file and do not match
the numbers Part 2 reports for the same model. That is unresolved as this is written, and the row is
marked in Section 5.3.

## 9. Future Work

The clearest next step comes from the error analysis and not from the scores. Three of the four
remaining errors are tweets describing more than one category, so this task really wants to be
multi label. Evaluating against multi label annotations would measure something the current setup
cannot.

The stress tests separate the models where the test metric does not, so they should be part of
choosing a model instead of a check run at the end. Picking a configuration by masked macro F1, or
by the gap between masked and randomly masked, would optimise for what the challenge actually asks
for. Training on partly masked text, so the model cannot lean on the strongest words, is the
obvious thing to try next. It tests whether the shortcut can be taken away by changing the training
signal rather than the architecture.

Three smaller things would tighten the comparison as it stands. An unweighted BiLSTM run would
remove the class weight difference and leave architecture as the only thing that changed. Running
DistilBERT with several seeds would let its comparison with the TextCNN sit against a seed spread.
Testing pretrained word embeddings in the BiLSTM and TextCNN would separate what pretraining gives
from what the architecture gives, which this design cannot do as it is.

## References for this section (IEEE)

This list continues the numbering used in Part 3. Entries [2] and [7] are already in that list and
are repeated here only so the citation can be identified. The numbering across Part 1, Part 3 and
Part 4 has to be merged into one list before submission, and each entry should be checked against
the source.

[2] Y. Kim, "Convolutional neural networks for sentence classification," in *Proc. 2014 Conf.
Empirical Methods in Natural Language Processing (EMNLP)*, Doha, Qatar, 2014, pp. 1746-1751.
(already in the list)

[7] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf.
Learning Representations (ICLR)*, 2015. (already in the list)

[12] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT:
smaller, faster, cheaper and lighter," 2019, arXiv:1910.01108.

[13] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional
transformers for language understanding," in *Proc. 2019 Conf. North American Chapter of the Assoc.
for Computational Linguistics (NAACL-HLT)*, Minneapolis, MN, USA, 2019, pp. 4171-4186.

[14] T. Wolf *et al.*, "Transformers: State-of-the-art natural language processing," in *Proc. 2020
Conf. Empirical Methods in Natural Language Processing: System Demonstrations*, 2020, pp. 38-45.

[15] A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in
*Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019, pp. 8024-8035.

[16] T. M. Pham, T. Bui, L. Mai, and A. Nguyen, "Out of order: How important is the sequential order
of words in a sentence in natural language understanding tasks?," in *Findings of the Assoc. for
Computational Linguistics: ACL-IJCNLP 2021*, 2021, pp. 1145-1160.

[17] K. Sinha, R. Jia, D. Hupkes, J. Pineau, A. Williams, and D. Kiela, "Masked language modeling
and the distributional hypothesis: Order word matters pre-training for little," in *Proc. 2021 Conf.
Empirical Methods in Natural Language Processing (EMNLP)*, 2021, pp. 2888-2913.

[18] R. Geirhos *et al.*, "Shortcut learning in deep neural networks," *Nature Machine
Intelligence*, vol. 2, no. 11, pp. 665-673, Nov. 2020.

[19] S. Gururangan, S. Swayamdipta, O. Levy, R. Schwartz, S. R. Bowman, and N. A. Smith,
"Annotation artifacts in natural language inference data," in *Proc. 2018 Conf. North American
Chapter of the Assoc. for Computational Linguistics (NAACL-HLT)*, New Orleans, LA, USA, 2018,
pp. 107-112.

[20] R. T. McCoy, E. Pavlick, and T. Linzen, "Right for the wrong reasons: Diagnosing syntactic
heuristics in natural language inference," in *Proc. 57th Annu. Meeting of the Assoc. for
Computational Linguistics (ACL)*, Florence, Italy, 2019, pp. 3428-3448.

[21] T. Niven and H.-Y. Kao, "Probing neural network comprehension of natural language arguments,"
in *Proc. 57th Annu. Meeting of the Assoc. for Computational Linguistics (ACL)*, Florence, Italy,
2019, pp. 4658-4664.
