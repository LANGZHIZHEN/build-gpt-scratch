## ch2. 数据准备
1.将文本转为单词序列，然后生成token:id索引
2.构建词转换encode和decode
3.滑动窗口采样，本质上是predict next word, 需要构建x和y，其中x和y维度相同，但是y向右偏移1位采样
4.编写dataloader，便于batch和shuffle，数据内置为list
5.编写词嵌入层，分别有token_embedding和position_embedding（torch.arrange实现）
6.token_embedding+position_embedding形成了我们的input_embedding

