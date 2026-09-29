# Resources

A curated list of external resources that pair well with this course. Use them as companions — not replacements. The course content is the spine; these are the muscle.

Sections below: math foundations, general ML, each algorithm family, deep learning, and a "how to read papers" section at the end for when you graduate from the course.

---

## Math foundations

### Linear algebra

- **3Blue1Brown — Essence of Linear Algebra** (YouTube, 15 videos). The single best visual intuition for vectors, matrices, determinants, eigen-things. Watch before module 00.
  - https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
- **Gilbert Strang — 18.06 Linear Algebra** (MIT OCW). The full course. Lectures 1–21 cover what you need; lectures 22–30 (eigenvalues, positive definite) are directly relevant to module 00 and module 09.
  - https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/
- **Immersive Linear Algebra** (interactive online book). The first immersive linear algebra textbook — manipulates vectors and matrices in the browser. Excellent for intuition.
  - https://immersivemath.com/ila/
- **StatLect — Linear Algebra**. Clean, concise reference entries for the specific operations used in ML (norms, projection, eigendecomposition, SVD).
  - https://www.statlect.com/matrix-algebra/

### Calculus

- **3Blue1Brown — Essence of Calculus** (YouTube, 11 videos). Intuition for derivatives, the chain rule, integrals. Watch before module 00.
  - https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr
- **Khan Academy — Multivariable Calculus**. The gradient, partial derivatives, and vector calculus you need for backprop. Sections on partial derivatives, gradient, and the chain rule are the ones to hit.
  - https://www.khanacademy.org/math/multivariable-calculus
- **Paul's Online Math Notes — Calculus III**. Concise reference for partial derivatives, gradients, and the chain rule in multiple variables.
  - https://tutorial.math.lamar.edu/Classes/CalcIII/CalcIII.aspx

### Probability & statistics

- **StatQuest — Statistics Fundamentals** (YouTube). Intuitive, visual explanations of distributions, expectation, variance, MLE, hypothesis testing. Good companion to module 00's probability section.
  - https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLYIcyhQ6nSqtU7g5z
- **Harvard Stat 110 — Probability** (Joe Blitzstein, YouTube / OCW). The full introductory probability course. Lectures on conditional probability, random variables, expectation, and common distributions are the most relevant.
  - https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo
- **Book: *Introduction to Probability* — Blitzstein & Hwang**. The Stat 110 textbook. Clear, example-rich, the best intro probability book for self-study.
  - https://www.amazon.com/Introduction-Probability-David-Blitzstein/dp/1466575572

---

## General machine learning

### Books

- ***An Introduction to Statistical Learning*** (ISL) — James, Witten, Hastie, Tibshirani. The most accessible ML book. R-based but the concepts are language-independent. Free PDF available. Covers everything through module 09 at a gentler pace than this course.
  - https://www.statlearning.com/ (free PDF)
- ***Elements of Statistical Learning*** (ESL) — Hastie, Tibshirani, Friedman. The graduate-level companion to ISL. The canonical reference for most of the algorithms in this course. Dense but authoritative.
  - https://hastie.su.domains/ElemStatLearn/ (free PDF)
- ***Pattern Recognition and Machine Learning*** (PRML) — Bishop. Bayesian-leaning, beautifully written. Covers modules 00–10 with a probabilistic framing. The math is cleaner than ESL in many places.
  - https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-PatternRecognitionAndMachineLearning-2006.pdf (free PDF)
- ***Deep Learning*** — Goodfellow, Bengio, Courville. The canonical deep learning reference. Covers modules 10–15 thoroughly. Free online.
  - https://www.deeplearningbook.org/

### Blogs & online

- **Distill.pub** — The best ML visualization journal. Articles are interactive, beautiful, and deep. Required reading for modules 09, 11, 12, 14.
  - https://distill.pub/
  - Standouts: *How to Use t-SNE Effectively* (Wattenberg et al., 2016), *The Building Blocks of Interpretability* (Olah et al., 2018), *Attention Is All You Need* visual explainer.
- **Jay Alammar — The Illustrated Transformer / BERT / Embedding**. The most accessible visual explanations of Transformers and related architectures. Read alongside modules 14 and 15.
  - https://jalammar.github.io/illustrated-transformer/
  - https://jalammar.github.io/illustrated-bert/
  - https://jalammar.github.io/illustrated-word2vec/
- **Lil'Log / Lilian Weng** — In-depth technical blog posts on RL, LLMs, attention, transfer learning. Higher difficulty than Alammar but excellent.
  - https://lilianweng.github.io/
- **Christopher Olah's blog** — Backpropagation calculus, RNNs, LSTM visual explanations. The "calculus on backpropagation" post is a must-read for module 10.
  - https://colah.github.io/posts/2015-08-Backprop/
  - https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **Sebastian Raschka — Machine Learning FAQ / Blog**. Practical explanations of ML concepts, often with code. Good for "how does this actually work" questions.
  - https://sebastianraschka.com/
- **Mike Bostock — Visualizing Algorithms** (and other visual essays). The gold standard for algorithmic visualization. Read for inspiration on how to think about diagrams (module 00 onward).
  - https://bost.ocks.org/mike/algorithms/
- **Papers With Code** — Browse by task, see the SOTA and the code. Useful for seeing what comes after this course.
  - https://paperswithcode.com/

### Video courses & channels

- **StatQuest with Josh Starmer** (YouTube). The single best ML explainer channel. Every algorithm in this course has a StatQuest video that gives you the intuition first. Watch these before each module for the intuition layer.
  - https://www.youtube.com/c/joshstarmer
- **Andrew Ng — Machine Learning** (Stanford CS229 / Coursera). The classic intro course. Covers the material of modules 00–09 at a higher level. Good for breadth after you've done the depth here.
  - https://www.coursera.org/learn/machine-learning
- **Andrew Ng — Deep Learning Specialization** (Coursera). Modules 10–13 covered at a practical level with TensorFlow. Good complement to the NumPy-from-scratch approach in this course.
  - https://www.coursera.org/specializations/deep-learning
- **fast.ai — Practical Deep Learning for Coders**. Top-down: get something working first, then understand why. The opposite philosophy from this course, which makes them perfect companions. Do courses 1 and 2 after finishing this one.
  - https://course.fast.ai/
- **Andrej Karpathy — Neural Networks / Let's build GPT / etc.** (YouTube). The best "from scratch" deep learning videos. His *Let's build GPT* video parallels module 15 almost exactly, and his *makemore* series is a great next step.
  - https://www.youtube.com/c/andrejkarpathy
  - https://www.youtube.com/watch?v=kCc8Fm1g1CE (Let's build GPT)

### Interactive

- **ML-From-Scratch** (Python library by Erik Linder-Norén). Full from-scratch implementations of many ML algorithms in NumPy — a good complement and sanity check for the implementations in this course.
  - https://github.com/eriklindernoren/ML-From-Scratch
- **The Neural Network Playground** (TensorFlow). Interactive visualization of a small NN training on toy data. Great for building intuition about hidden layers, activation functions, and learning rate before module 10.
  - https://playground.tensorflow.org/
- **Seeing Theory** (Brown University). Interactive visualizations for probability and statistics concepts.
  - https://seeing-theory.brown.edu/

---

## Algorithm-by-algorithm further reading

### Linear regression (module 01)

- **Galton's original regression paper** (1886). The historical origin of the term "regression." Short and readable.
  - https://www.stat.ox.ac.uk/~nbriggs/History.pdf
- **Ordinary Least Squares** — Encyclopedia of Mathematics. Concise formal reference.
  - https://encyclopediaofmath.org/wiki/Ordinary_least_squares

### Logistic regression (module 02)

- **Cox (1958)** — the original proportional hazards / logistic regression paper. The historical reference.
  - https://doi.org/10.1111/j.2517-6161.1958.tb00290.x
- **StatsModels docs — Logit**. Shows the GLM framing and the IRLS fitting procedure in practice.
  - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html

### Regularization (module 03)

- **Tibshirani (1996)** — the Lasso paper. The original.
  - https://arxiv.org/abs/1612.00601 (reprint)
- **Zou & Hastie (2005)** — ElasticNet. The original.
  - https://www.jstor.org/stable/4141867
- ** scikit-learn User Guide — Regularization**. Well-written practical summary of Ridge, Lasso, ElasticNet with examples.
  - https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-lasso

### SVM (module 04)

- **Cortes & Vapnik (1995)** — the canonical SVM paper.
  - https://link.springer.com/article/10.1007/BF00994018
- **Boser, Guyon & Vapnik (1992)** — the max-margin classifier that preceded SVM.
  - https://dl.acm.org/doi/10.1145/130385.130401
- **Burges (1998)** — *A Tutorial on Support Vector Machines for Pattern Recognition*. The best tutorial on SVMs. Long but complete — covers the geometry, the primal, the dual, kernels, and practical issues.
  - https://link.springer.com/chapter/10.1007/BFb0094183
- **Schölkopf & Smola — *Learning with Kernels*** (book). The definitive book on kernel methods. Covers SVM, kernel PCA, and the theory behind kernels.
  - https://www.microsoft.com/en-us/research/publication/learning-with-kernels/

### Decision trees (module 05)

- **Quinlan (1986)** — the ID3 paper. The origin of entropy-based tree splitting.
  - https://link.springer.com/article/10.1007/BF00116253
- **Breiman et al. (1984)** — *Classification and Regression Trees* (CART book). The definitive reference on tree theory.
  - https://www.amazon.com/Classification-Regression-Trees-Leo-Breiman/dp/0412048418
- ** scikit-learn docs — Decision Trees**. Practical reference with the splitting criteria and pruning strategies.
  - https://scikit-learn.org/stable/modules/tree.html

### Ensembles (module 06)

- **Breiman (1996)** — Bagging Predictors. The original bagging paper.
  - https://www.jstor.org/stable/2246110
- **Breiman (2001)** — Random Forests. The RF paper.
  - https://www.jstor.org/stable/2699409
- **Freund & Schapire (1997)** — *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting* (AdaBoost). The definitive AdaBoost paper.
  - https://dl.acm.org/doi/10.1006/jvlc.1997.0100
- **Friedman (2001)** — *Greedy Function Approximation: A Gradient Boosting Machine*. The GBM paper.
  - https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-Function-Approximation-A-Gradient-Boosting-Machine/10.1214/aos/1013203471.single
- **Chen & Guestrin (2016)** — *XGBoost: A Scalable Tree Boosting System*. The XGBoost paper.
  - https://arxiv.org/abs/1603.02754
- **Ke et al. (2017)** — *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. The LightGBM paper.
  - https://proceedings.neurips.cc/paper/2017/file/6907456f02ccfe89359415668c2cc3a1-Paper.pdf

### Naive Bayes & kNN (module 07)

- **Fix & Hodges (1951)** — early kNN work.
  - https://www.jstor.org/stable/2236906
- **Cover & Hart (1967)** — *Nearest Neighbor Pattern Classification*. The seminal kNN theoretical paper.
  - https://doi.org/10.1109/TIT.1967.1053964
- **Rennie et al. (2003)** — *Tackling the Poor Assumptions of Naive Bayes Text Classifiers*. Good discussion of why Naive Bayes works despite its wrong assumptions.
  - https://www.icml-2003.org/proceedings/345.pdf

### Clustering (module 08)

- **Lloyd (1982)** — *Least squares quantization in PCM*. The k-means algorithm paper.
  - https://ieeexplore.ieee.org/document/1100787
- **Arthur & Vassilvitskii (2007)** — *k-means++: The Advantages of Careful Seeding*. The k-means++ initialization paper.
  - https://dl.acm.org/doi/10.5555/1283383.1283494
- **Dempster, Laird & Rubin (1977)** — *Maximum Likelihood from Incomplete Data via the EM Algorithm*. The EM paper. A classic.
  - https://pubmed.ncbi.nlm.nih.gov/8238867/
- **Bishop PRML Chapter 9** — Mixture models and EM. The clearest textbook treatment of EM for GMM.
  - (see Books above)

### Dimensionality reduction (module 09)

- **Pearson (1901)** — the original PCA paper.
  - https://doi.org/10.1080/14786440109462720
- **Hotelling (1933)** — the modern PCA formulation.
  - https://doi.org/10.1037/h0070888
- **van der Maaten & Hinton (2008)** — *Visualizing Data using t-SNE*. The t-SNE paper.
  - https://www.jmlr.org/papers/v9/vandermaaten08a.html
- **McInnes, Healy & Melville (2018)** — *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. The UMAP paper.
  - https://arxiv.org/abs/1802.03426
- **Wattenberg, Viégas & Johnson (2016)** — *How to Use t-SNE Effectively*. Distill.pub. Required reading on t-SNE interpretation.
  - https://distill.pub/2016/misread-tsne/

### Neural networks / MLP (module 10)

- **Rosenblatt (1958)** — *The Perceptron*. The historical origin.
  - https://psycnet.apa.org/record/1959-04275-001
- **Rumelhart, Hinton & Williams (1986)** — *Learning Representations by Back-propagating Errors*. The backprop paper that popularized it.
  - https://www.nature.com/articles/323533a0
- **Goodfellow, Bengio & Courville — *Deep Learning***, Chapter 6 (Deep Feedforward Networks). The best textbook reference for MLPs.
  - (see Books above)
- **He et al. (2015)** — *Delving Deep into Rectifiers*. The He initialization paper.
  - https://arxiv.org/abs/1502.01852

### Training deep nets (module 11)

- **Robbins & Monro (1951)** — stochastic approximation, the SGD precursor.
  - https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.single
- **Kingma & Ba (2014)** — *Adam: A Method for Stochastic Optimization*. The Adam paper.
  - https://arxiv.org/abs/1412.6980
- **Srivastava et al. (2014)** — *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*.
  - https://www.jmlr.org/papers/v15/srivastava14a.html
- **Ioffe & Szegedy (2015)** — *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*.
  - https://arxiv.org/abs/1502.03167
- **Loshchilov & Hutter (2016)** — *SGDR: Stochastic Gradient Descent with Warm Restarts*. Cosine annealing.
  - https://arxiv.org/abs/1608.03983
- **Smith (2015)** — *Cyclical Learning Rates for Training Neural Networks*.
  - https://arxiv.org/abs/1506.01186

### CNNs (module 12)

- **Hubel & Wiesel (1962)** — the neuroscience paper that inspired CNNs.
  - https://doi.org/10.1113/jphysiol.1962.sp006837
- **LeCun et al. (1998)** — *Gradient-Based Learning Applied to Document Recognition*. LeNet-5.
  - https://doi.org/10.1109/5.726791
- **Krizhevsky, Sutskever & Hinton (2012)** — *ImageNet Classification with Deep Convolutional Neural Networks*. AlexNet.
  - https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c843688beede52a3-Paper.pdf
- **Goodfellow et al. — *Deep Learning***, Chapter 9 (Convolutional Networks).
  - (see Books above)
- **Stanford CS231n — Convolutional Neural Networks for Visual Recognition**. The best CNN course. Lectures and notes are freely available and pair well with module 12.
  - http://cs231n.stanford.edu/

### RNNs & LSTMs (module 13)

- **Elman (1990)** — *Finding Structure in Time*. The simple RNN paper.
  - https://doi.org/10.1207/s15516709cog1402_1
- **Hochreiter & Schmidhuber (1997)** — *Long Short-Term Memory*. The LSTM paper. Required reading.
  - https://doi.org/10.1162/neco.1997.9.8.1735
- **Bengio et al. (1994)** — *Learning Long-Term Dependencies with Gradient Descent is Difficult*. The vanishing gradient analysis.
  - https://ieeexplore.ieee.org/document/5548508
- **Goodfellow et al. — *Deep Learning***, Chapter 10 (RNNs).
  - (see Books above)
- **Graves (2012)** — *Supervised Sequence Labelling with Recurrent Neural Networks*. The book.
  - https://www.cs.toronto.edu/~graves/torch5rnnlib.pdf

### Attention (module 14)

- **Bahdanau, Cho & Bengio (2015)** — *Neural Machine Translation by Jointly Learning to Align and Translate*. The additive attention paper. The first use of attention in NMT.
  - https://arxiv.org/abs/1409.0473
- **Luong, Pham & Manning (2015)** — *Effective Approaches to Attention-based Neural Machine Translation*. Global vs local attention, the scaled dot-product precursor.
  - https://arxiv.org/abs/1508.04025

### Transformers (module 15)

- **Vaswani et al. (2017)** — *Attention Is All You Need*. THE paper. Everything in module 15 traces back to this.
  - https://arxiv.org/abs/1706.03762
- **Devlin et al. (2019)** — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
  - https://doi.org/10.18653/v1/N19-1423
- **Radford et al. (2018)** — *Improving Language Understanding by Generative Pre-Training*. GPT-1.
  - https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- **Radford et al. (2019)** — *Language Models are Unsupervised Multitask Learners*. GPT-2.
  - https://cdn.openai.com/blogs/community-gpt2/model-card-GBT2.pdf
- **Brown et al. (2020)** — *Language Models are Few-Shot Learners*. GPT-3.
  - https://arxiv.org/abs/2005.14165
- **Su et al. (2021)** — *RoFormer: Enhanced Transformer with Rotary Position Embedding*. The RoPE paper.
  - https://arxiv.org/abs/2104.09864
- **Dai et al. (2019)** — *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*.
  - https://arxiv.org/abs/1901.02860
- **Kaplan et al. (2020)** — *Scaling Laws for Neural Language Models*.
  - https://arxiv.org/abs/2001.08361
- **Hoffmann et al. (2022)** — *Training Compute-Optimal Large Language Models (Chinchilla)*.
  - https://arxiv.org/abs/2203.15556
- **Stanford CS224N — NLP with Deep Learning**. The best NLP course. Lectures on Transformers, attention, BERT, and GPT are the best companions to modules 14 and 15.
  - http://web.stanford.edu/class/cs224n/
- **Google — The Transformer: A Novel Architecture for Language Understanding** (blog post). The official visual explainer of the Transformer architecture.
  - https://ai.googleblog.com/2017/08/transformer-novel-architecture-for.html

---

## Lab companions

### Lab A — Classical Shootout

- **Wolpert & Macready (1997)** — *No Free Lunch Theorems for Optimization*. The theoretical result that no algorithm is universally best.
  - https://ieeexplore.ieee.org/document/613152

### Lab B — PCA vs t-SNE vs UMAP

- **Tenenbaum, de Silva & Langford (2000)** — *A Global Geometric Framework for Nonlinear Dimensionality Reduction*. Isomap — the manifold learning method that inspired much of the field.
  - https://www.science.org/doi/10.1126/science.290.5500.2319
- **Roweis & Saul (2000)** — *Nonlinear Dimensionality Reduction by Locally Linear Embedding*. LLE.
  - https://www.science.org/doi/10.1126/science.290.5500.2323

### Lab C — MLP vs CNN

- **AlexNet paper** — see CNNs section above. The paper that made CNNs the default for vision.

### Lab D — Attention from Scratch

- **Dao et al. (2022)** — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. The engineering paper that made long context windows practical.
  - https://arxiv.org/abs/2205.14135
- **Vaswani et al. (2017)** — the primary reference. See Transformers section above.

---

## How to read ML papers

You'll start reading papers in earnest after module 04. Here's a method that works:

1. **Read the abstract and the introduction.** Stop. Write down: what problem are they solving, what's the key idea, and what's the claim.
2. **Skim the related work section.** Map the paper onto what you already know from this course. Which module does this connect to?
3. **Read the method section slowly.** Derive the key equation on paper. If you can't follow a step, look up the notation — don't skip it.
4. **Read the experiments section.** What dataset, what baseline, what metric? Are the comparisons fair?
5. **Re-derive the key result from the course content.** If the paper extends an algorithm from this course (e.g., generalizes logistic regression, or adds attention to an RNN), write down exactly what changed relative to the course version.

Papers worth reading first (in order):

1. **Vaswani et al. (2017)** — *Attention Is All You Need*. The most important paper in modern ML. Read it after module 14. You'll understand most of it by then.
2. **Bahdanau et al. (2015)** — the attention paper that preceded the Transformer. Shorter, easier. Read before Vaswani.
3. **Hochreiter & Schmidhuber (1997)** — the LSTM paper. Read after module 13.
4. **Kingma & Ba (2014)** — the Adam paper. Short, readable, directly relevant to module 11.

A good habit: after finishing each module, pick one paper from that module's "Papers" section and read it. By the end of the course, you'll have read 16 papers and you'll know how to read the rest.

---

## Gen-AI companion course

This course is the theory track. For the engineering track — building with LLMs, RAG, agents, fine-tuning, multimodal, deployment — see:

- **[gen-ai](https://github.com/Lourdhu02/gen-ai)** — the practical companion course by the same author.

Do gen-ai after finishing this course, or in parallel once you reach module 15.

---

## Contribute a resource

Found a resource that's missing? Open an issue or PR. Good additions:

- A blog post that explains one of the course algorithms better than anything listed here.
- A video that gives exceptional intuition for a specific topic.
- A paper that's essential for one of the modules and isn't listed.
- A free course or textbook that covers a topic in this course at the right level.

Please keep the list curated — one or two best resources per topic, not a dump of everything available.
