---
title: tensorflow Word2Vec教程代码解读
date: 2017-06-04 23:28:28
tags:
- nlp
- embedding
categories:
- 深度学习与智能
---

本文主要解读tensorflow在[这个教程](https://github.com/tensorflow/models/tree/master/tutorials/embedding)中的词嵌入(embedding)模块代码.

### 代码解读
- 教程中的`word2vec.py`是主要的词嵌入模块.`word2vec_optimized.py`是其优化版本,主要是将代价函数部分用C++实现了,提高了训练效率.这两个版本的模块都使用了C++实现的自定义读取操作,需要在运行前先编译出该操作的so库.
- 代码强制指定了各项计算都使用CPU,这是有道理的,见文末的讨论.
- 解读以 `word2vec.py` 为主。

#### overview
- 使用`tf.app`来运行整个模块，以便使用`tf.app.flag`来接收和管理传入的参数。
- 接收的训练数据是连续的句子语料文本。使用C++自定义文本处理操作(也就是op),负责将连续的句子转换为符合skip模型的样本.
- `Class Options`用来管理配置和参数.
- `Class Word2Vec`是主体，包含主要逻辑。
- 使用`thread`库来进行多线程训练.

#### 训练

##### forward(self, examples, labels)
这个函数建立前向网络。输入的`examples`和`labels`分别是一个batch的训练样本和标签的tensor占位符,由预处理操作提供.函数返回输入给激活函数的logits.
首先初始化各权值矩阵.`emb`是我们需要的词嵌入字典,尺寸是`单词数` * `emb_dim`，然后逻辑回归网络仅有一层,其权值矩阵是`sm_w_t`('t'表示转置)和`sm_b`：

```python
	# Declare all variables we need.
    # Embedding: [vocab_size, emb_dim]
    init_width = 0.5 / opts.emb_dim
    emb = tf.Variable(
        tf.random_uniform(
            [opts.vocab_size, opts.emb_dim], -init_width, init_width),
        name="emb")
    self._emb = emb

    # Softmax weight: [vocab_size, emb_dim]. Transposed.
    sm_w_t = tf.Variable(
        tf.zeros([opts.vocab_size, opts.emb_dim]),
        name="sm_w_t")

    # Softmax bias: [emb_dim].
    sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")
```
注意这里`sm_w_t`是输出层权值矩阵.由于我们要输出的label数量等于单词表中单词的数量,所以这里矩阵的维度恰好是`单词数` * `emb_dim`,恰好与embedding矩阵的尺寸相同.
第一步是进行噪声采样，就是从所有词中随机选出若干词作为噪声。使用了`tf.nn.fixed_unigram_candidate_sampler`采样函数
```python
    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(
        tf.cast(labels,
                dtype=tf.int64),
        [opts.batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=opts.num_samples,
        unique=True,
        range_max=opts.vocab_size,
        distortion=0.75,
        unigrams=opts.vocab_counts.tolist()))
```
然后，使用`tf.nn.embedding_lookup`函数从权值矩阵中选出对应的权值组成小的权值矩阵。这一步实际上是等效于以one-hot形式的输入数据乘以2层隐藏层1层输出层的权值矩阵。由于one-hot形式非常稀疏，我们仅选出第一层中对应的权值（其他权值在计算中都归零了）；同时由于我们仅关心真实词和采样噪声词的预测结果，所以仅选出第二层(输出层)中的对应权值：
```python
    # Embeddings for examples: [batch_size, emb_dim]
    example_emb = tf.nn.embedding_lookup(emb, examples)

    # Weights for labels: [batch_size, emb_dim]
    true_w = tf.nn.embedding_lookup(sm_w_t, labels)
    # Biases for labels: [batch_size, 1]
    true_b = tf.nn.embedding_lookup(sm_b, labels)

    # Weights for sampled ids: [num_sampled, emb_dim]
    sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
    # Biases for sampled ids: [num_sampled, 1]
    sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)
```
计算并返回logit：
```python
    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise labels for all examples in the batch
    # using the matmul.
    sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
    sampled_logits = tf.matmul(example_emb,
                               sampled_w,
                               transpose_b=True) + sampled_b_vec
    return true_logits, sampled_logits

```
这里有个细节.注意到对于一个batch的样本而言,正确的词(标签)每个样本对应一个,一共有batch个,计算时对于每个样本而言是两个向量相乘(使用`tf.multiply`)再加上偏置.而对于抽到的负样本而言,一个batch的样本都使用同样的simple_num个干扰样本,对于每个样本的计算都是一个向量(样本的词向量)乘以一个矩阵(simple_num个噪声样本对应的权值向量拼接起来),使用`tf.matmul`.所以可以看到计算方式是不同的.

##### nce_loss(self, true_logits, sampled_logits)

训练中实际用到的代价函数.NCE代价函数简单来说就是把标签词的交叉熵与干扰词的交叉熵相加得到的结果.我们指定标签词应该得到1,干扰词应该得到0,这样当标签词的activation越接近1,交叉熵就越小;当干扰词的activation越接近0,其交叉熵也越小.求两者相加的NCE代价越小,就等价于使模型结果尽量倾向于标签词而远离干扰词.
代码首先使用`tf.nn.sigmoid_cross_entropy_with_logits`函数计算标签词和干扰词的交叉熵.该函数的`labels`参数是一个维度与输入`logits`相同的tensor,指定了对应位置上的logit应该被判断为1还是0：
```python
    # cross-entropy(logits, labels)
    opts = self._options
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(true_logits), logits=true_logits)
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

```
然后,将标签词的交叉熵与噪音词的交叉熵相加,并且在`batch_size`上做平均,就得到了NCE:

```python

    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (tf.reduce_sum(true_xent) +
                       tf.reduce_sum(sampled_xent)) / opts.batch_size
    return nce_loss_tensor

```

##### build_graph(self)
建立训练网络.通过组合前向网络和优化器,得到最终的训练tensor.
首先,通过C++自定义的`word2vec.skipgram_word2vec`操作得到输入数据的tensor,这些样本将符合skipgram模型.关于skipgram模型的原理这里不再介绍,符合该模型的输入样本是一个词(id),其对应的标签也是一个词(id).每个样本表示出现该词时,其上下文(一个指定的窗口范围内)有可能出现如标签所标记的词.
这个操作将返回若干统计量和包含输入数据以及标签的占位符.这些返回参数都是tensor,需要通过`session.run`才能获取到结果值.

```python
# The training data. A text file.
    #处理数据读取过程.把文本按照skip模型的方式分割成一个个样本
    #这里返回的参数都是placeholder,需要通过session.run才能填充实际数据
    #word:包含所有单词的向量
    #counts:每个单词在语料中出现的频数
    #words_per_epoch:每个epoch中包含的单词数目
    #self._epoch:当前epoch序号
    #self._words:当前已处理单词数
    #examples:样本词id(一个batch大小)
    #labels:标签词id(一个batch大小)
    (words, counts, words_per_epoch, self._epoch, self._words, examples,
     labels) = word2vec.skipgram_word2vec(filename=opts.train_data,
                                          batch_size=opts.batch_size,
                                          window_size=opts.window_size,
                                          min_count=opts.min_count,
                                          subsample=opts.subsample)
    #获取单词表和频数统计
    (opts.vocab_words, opts.vocab_counts,
     opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
    opts.vocab_size = len(opts.vocab_words)
    print("Data file: ", opts.train_data)
    print("Vocab size: ", opts.vocab_size - 1, " + UNK")
    print("Words per epoch: ", opts.words_per_epoch)
    self._examples = examples
    self._labels = labels
    self._id2word = opts.vocab_words
    #反向查询表
    for i, w in enumerate(self._id2word):
      self._word2id[w] = i
```
完成对语料的预处理后,将前向过程和代价函数组合起来,使用优化器自动进行反向过程.最后定义了一个记录所有tensor状态的`saver`

```python
    #前向计算标签和干扰项的logits
    true_logits, sampled_logits = self.forward(examples, labels)
    #根据上述logits计算nce
    loss = self.nce_loss(true_logits, sampled_logits)
    tf.summary.scalar("NCE loss", loss)
    self._loss = loss
    #修正权重
    self.optimize(loss)

    # Properly initialize all variables.
    tf.global_variables_initializer().run()

    self.saver = tf.train.Saver()
```

##### train(self)
实际执行训练过程.这里使用了`thread`模块来进行多线程训练.代码仅进行了一次epoch就停机了,实际使用时增加epoch数目也许可以获得更好的结果.
真正执行训练的代码在子线程函数中:
```python
def _train_thread_body(self):
    #获取当前epoch序号
    initial_epoch, = self._session.run([self._epoch])
    while True:
      #每次执行都是执行一个batch.当处理完一次epoch的单词之后,就退出这个线程
      _, epoch = self._session.run([self._train, self._epoch])
      if epoch != initial_epoch:
        break
```
这个函数首先run一次`epoch`tensor获取当前处理的epoch.之后通过`session.run`计算`train`tensor,这就进行了前向和反向操作,同时也获取了新的`epoch`值.当这个值与初始epoch序号不同时,就终止了训练.
这个子线程通过`thread`库进行了调用:

```python
    workers = []
    for _ in xrange(opts.concurrent_steps):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)
```
训练线程启动之后,代码进入了一个循环,将不断run那些统计tensor以打印当前的状态,并定时保存checkpoint.在`epoch`的值增加之后(跑完了一遍epoch)就等待各线程结束:

```python
    #进行循环统计信息
    while True:
      time.sleep(opts.statistics_interval)  # Reports our progress once a while.
      #算一次前向过程以提供统计数据
      (epoch, step, loss, words, lr) = self._session.run(
          [self._epoch, self.global_step, self._loss, self._words, self._lr])
      now = time.time()
      #计算相关统计参数并打印
      last_words, last_time, rate = words, now, (words - last_words) / (
          now - last_time)
      print("Epoch %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r" %
            (epoch, step, lr, loss, rate), end="")
      sys.stdout.flush()
      #记录信息
      if now - last_summary_time > opts.summary_interval:
        summary_str = self._session.run(summary_op)
        summary_writer.add_summary(summary_str, step)
        last_summary_time = now
      #保存checkpoint
      if now - last_checkpoint_time > opts.checkpoint_interval:
        self.saver.save(self._session,
                        os.path.join(opts.save_path, "model.ckpt"),
                        global_step=step.astype(int))
        last_checkpoint_time = now
      #处理完整一个epoch之后就退出统计,并且等待各计算线程工作完毕
      if epoch != initial_epoch:
        break
```

### 一些问题
- 在我的机器上，默认代码虽然占用了gpu显存但是并不会真正利用gpu。强行修改代码利用gpu的话，速度比cpu还慢。待解决。
- 目前看来，代码使用cpu(默认)就是比gpu快速。打印tensor分配日志可以看到部分op被分配到了cpu上。实际上，自定义实现的op都没有gpu版本，这包括了数据IO操作，和optimized版本代码中的损失函数操作。由于embedding可能很大，所以这个矩阵不应该被分配到gpu上；同时，embedding lookup和embedding update都不会被分配到gpu上。
- 目前观察到现象是，无论是`word2vec.py`还是`word2vec_optimized.py`，都是指定分配任务到cpu上的性能更好，而如果不强制指定cpu，那么`word2vec_optimized.py`性能没有太大变化，而`word2vec.py`性能严重下降。两者的差异在于，`word2vec_optimized.py`自行实现了nce损失函数的op（没有gpu代码）。推测出现性能差异的原因是，`word2vec.py`的 embedding update 操作在cpu上而损失函数（以及隐藏层更新）在gpu上，这种操作带来了严重的通信开销；而`word2vec_optimized.py`的损失函数也在cpu上（因为自定义op没有实现gpu版本），所以损失不大。也可以指定仅将matmul操作分配到gpu上，实测性能好于让机器自行分配，但是仍然比指定全cpu性能差。
- 由此可以得出，word embeding最好在cpu上做。
