## ch2. 数据准备
1.将文本转为单词序列，然后生成token:id索引 <br>
2.构建词转换encode和decode <br>
3.滑动窗口采样，本质上是predict next word, 需要构建x和y，其中x和y维度相同，但是y向右偏移1位采样 <br>
4.编写dataloader，便于batch和shuffle，数据内置为list <br>
5.编写词嵌入层，分别有token_embedding和position_embedding（torch.arrange实现）<br>
6.token_embedding+position_embedding形成了我们的input_embedding <br>
## ch2.注意力模型构建
1.计算注意力分数，query与key的点积，使用矩阵乘法优化
2.计算注意力权重，softmax(dim=-1)
3.计算上下文向量，表示对于输入词元所有信息的聚合

