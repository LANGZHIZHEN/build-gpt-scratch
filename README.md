# 二周目探索

# 一周目探索
## ch7.指令任务微调
1.准备数据集，编写Dataset，collate函数，loader <br>
2.加载预训练权重gpt2-medium<br>
3.训练后，下载ollama，部署qwen3-4B对test_data进行评估打分（本地进行）<br>
4.测试ollama，使用gemma3:4b对模型答复进行打分<br>
5.一周目结束，顺便把代码值得复用的部分重写并结构化
## ch6.分类任务微调
1.准备数据集，保存成csv格式（主要是分好input和target） <br>
2.编写Dataset类，将text转为词元id，包括每个item的长度保证统一，不足则添加<|endoftext|>；编写Dataloader类<br>
3.确定微调的参数层，这里只修改了最后一个transformer块和层归一化，然后修改了输出的二分类<br>
4.编写了交叉熵损失函数以及训练流程<br>
5.编写评估函数，计算loss和accuracy<br>
6.可视化训练loss和acc<br>
## ch5.训练GPT 
1.使用CrossEntroyLoss函数，AdamW优化器进行训练 <br>
2.编写计算loader损失和batch损失函数，分别用于评估和训练<br>
3.编写整个训练过程<br>
4.设计温度解码策略和top-k选择<br>
5.保存模型，加载<br>
6.下载预训练权重gpt2-124M, main.py中添加训练和推理两部分<br>
## ch4.搭建GPT的模型
1.编写层归一化模型，注意分母添加eps防止NaN <br>
2.编写前馈神经网络，用GELU代替RELU <br>
3.编写残差连接，注意nn.Sequential和nn.ModuleList的区别 <br>
4.编写TransformerBlock，包含上述模块。<br>
5.编写GPTModel，包含TransformerBlock的堆叠和embedding层<br>
5.编写后处理函数，提取最后一行向量的最可能词元并转为ID，输出<br>
## ch3.注意力模型构建
1.计算注意力分数，query与key的点积，使用矩阵乘法优化 <br>
2.计算注意力权重，softmax(dim=-1) <br>
3.计算上下文向量，表示对于输入词元所有信息的聚合 <br>
4.添加掩码，对权重进行mask，只能访问当前位置之前的词；添加dropout正则化， <br>
5.添加多头层，将output_dim拆分为num_heads * head_dim, 针对每一个head_dim进行计算，最后在out_proj里投影映射。<br>
6.pytorch的矩阵乘法默认对向量的后两个维度进行乘法，所以在batch中需要对向量进行transpose；维度拆分和合并时则需要views整理维度。
## ch2. 数据准备
1.将文本转为单词序列，然后生成token:id索引 <br>
2.构建词转换encode和decode <br>
3.滑动窗口采样，本质上是predict next word, 需要构建x和y，其中x和y维度相同，但是y向右偏移1位采样 <br>
4.编写dataloader，便于batch和shuffle，数据内置为list <br>
5.编写词嵌入层，分别有token_embedding和position_embedding（torch.arrange实现）<br>
6.token_embedding+position_embedding形成了我们的input_embedding <br>



