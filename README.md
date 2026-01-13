## ch2. 数据准备
1.将文本转为单词序列，然后生成token:id索引 <br>
2.构建词转换encode和decode <br>
3.滑动窗口采样，本质上是predict next word, 需要构建x和y，其中x和y维度相同，但是y向右偏移1位采样 <br>
4.编写dataloader，便于batch和shuffle，数据内置为list <br>
5.编写词嵌入层，分别有token_embedding和position_embedding（torch.arrange实现）<br>
6.token_embedding+position_embedding形成了我们的input_embedding <br>
## ch2.注意力模型构建
1.计算注意力分数，query与key的点积，使用矩阵乘法优化 <br>
2.计算注意力权重，softmax(dim=-1) <br>
3.计算上下文向量，表示对于输入词元所有信息的聚合 <br>
4.添加掩码，对权重进行mask，只能访问当前位置之前的词；添加dropout正则化， <br>
5.添加多头层，将output_dim拆分为num_heads * head_dim, 针对每一个head_dim进行计算，最后在out_proj里投影映射。<br>
6.pytorch的矩阵乘法默认对向量的后两个维度进行乘法，所以在batch中需要对向量进行transpose；维度拆分和合并时则需要views整理维度。

