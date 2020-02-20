# 文本表示的方法
下面对文本表示进行一个归纳，也就是对于一篇文本可以如何用数学语言表示呢？
- 基于one-hot、朴素贝叶斯、tf-idf、text rank等的bag-of-words；
	1. 词袋（bag of words）模型：采用词袋模型（即计算文章中各个单词出现的次数）来建立特征输入机器学习分类器。
	1. 关键词提取方法：tf-idf、text rank
- 主题模型：LSA（SVD）、pLSA、LDA；
- 基于词向量的固定表征：word2vec、fastText、glove；
- 基于词向量的动态表征：ELMo、GPT、Transformer、Bert、XLNet、ALBERT；

# 语言模型发展历史
第一代：N-gram(马尔科夫链，统计学)--->第二代：BOW/LDA(统计学语言模型)--->第三代：word2vec(CBOW/Skip-gram，浅层神经网络)--->第四代：ELMo、GPT、Bert(深度学习时代)


# NLP经典论文
1. A neural probabilistic language model    
Yoshua Bengio · Rejean Ducharme · Pascal Vincent · Christian Janvin    
2003 · Journal of Machine Learning Research | 被引数：3297     
A goal of statistical language modeling is to learn the joint probability function of sequences of words in a language. This is intrinsically difficult because of the curse of dimensionality: a word sequence on which the model will be tested is likely to be different from all the word sequences seen during training. Traditional but very successful approaches based on n-grams obtain generalization by concatenating very short overlapping sequences seen in the training set. We propose to fight the curse of dimensionality by learning a distributed representation for words which allows each training sentence to inform the model about an exponential number of semantically neighboring sentences. The model learns simultaneously (1) a distributed representation for each word along with (2) the probability function for word sequences, expressed in terms of these representations. Generalization is obtained because a sequence of words that has never been seen before gets high probability if it is made of words that are similar (in the sense of having a nearby representation) to words forming an already seen sentence. Training such large models (with millions of parameters) within a reasonable time is itself a significant challenge. We report on experiments using neural networks for the probability function, showing on two text corpora that the proposed approach significantly improves on state-of-the-art n-gram models, and that the proposed approach allows to take advantage of longer contexts.     


1. Natural Language Processing (Almost) from Scratch   
Ronan Collobert · Jason Weston · Leon Bottou · Michael Karlen · Koray Kavukcuoglu · Pavel P Kuksa   
2011 · Journal of Machine Learning Research | 被引数：3384     
We propose a unified neural network architecture and learning algorithm that can be applied to various natural language processing tasks including part-of-speech tagging, chunking, named entity recognition, and semantic role labeling. This versatility is achieved by trying to avoid task-specific engineering and therefore disregarding a lot of prior knowledge. Instead of exploiting man-made input features carefully optimized for each task, our system learns internal representations on the basis of vast amounts of mostly unlabeled training data. This work is then used as a basis for building a freely available tagging system with good performance and minimal computational requirements.  

1. Attention Is All You Need       
ELMo等基于深度学习的方法可以有效地学习出上下文有关词向量，但毕竟是基于LSTM的序列模型，即便结合使用注意力机制，必然也要面临梯度以及无法并行化的问题。在本文中，我们重点来讲解Transformer模型，它的核心是Self-Attention机制，并且Transformer也是BERT的核心组成部分。

1. 从Transformer到BERT模型        
在18年年底的时候，有一件事情轰动了整个NLP界，它就是大家所熟悉的BERT模型，它刷新了整个文本领域的排行榜，受到了全球的瞩目。之后，很多公司慢慢开始采用BERT作为各种应用场景的预训练模型来提高准确率。在本文里，我们重点来讲解BERT模型以及它的内部机制。
