# 文本表示的方法
下面对文本表示进行一个归纳，也就是对于一篇文本可以如何用数学语言表示呢？
- 基于one-hot、tf-idf、textrank等的bag-of-words；
- 主题模型：LSA（SVD）、pLSA、LDA；
- 基于词向量的固定表征：word2vec、fastText、glove；
- 基于词向量的动态表征：elmo、GPT、bert；

# 语言模型发展历史
第一代：N-gram(马尔科夫链，统计学)--->第二代：BOW/LDA(统计学语言模型)--->第三代：word2vec(CBOW/Skip-gram，浅层神经网络)--->第四代：ELMo、GPT、Bert(深度学习时代)