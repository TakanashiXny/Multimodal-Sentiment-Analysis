# Multimodal Sentiment Analysis

## 运行环境

运行
```
cd Multimodal-Sentiment-Analysis
pip install -r requirements.txt
```

关于pytorch版本：此次实验在AutoDL云服务器上进行，运行指令`pip show torch`得到如下信息

Name: torch

Version: 1.11.0+cu113

Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration

Home-page: https://pytorch.org/

Author: PyTorch Team

Author-email: packages@pytorch.org

License: BSD-3

Location: /root/miniconda3/lib/python3.8/site-packages

Requires: typing-extensions

Required-by: torchvision

本次实验使用A40卡完成

## 代码完整结构

**需要注意**，由于模型文件和数据文件较大，没有上传至github，仅仅通过邮件发送，所以github上没有data和model两个文件夹

```python
|-- data # 这个文件夹包含了所有需要使用的数据
    |-- images # 这个文件夹中包含了所有需要使用的图片数据
    |-- test_text.txt # 这个文件中包含了所有需要使用的测试数据
    |-- train_text.csv # 这个文件中包含了整理完成的训练数据
|-- model # 这个文件夹中包含了需要使用的预训练模型
    |-- bert_tokenizer # 这个文件夹中包含了bert分词器
        |-- config.json # 分词配置信息
        |-- tokenizer.json # 分词器
    |-- bert-base-uncased # bert模型
        |-- config.json # 模型配置信息
        |-- pytorch_model.bin # 模型内容
    |-- resnet50 # resnet模型
        |-- config.json # 模型配置信息
        |-- pytorch_model.bin # 模型内容
    |-- model.pt # 用于预测的模型
|-- util # 这个文件夹包含了数据预处理的代码
    |-- data_process.py # 数据处理代码
|-- .gitignore # 不记录的文件
|-- README.md # 本项目的介绍
|-- main.py # 主函数
|-- model.py  # 构建多模态模型的代码
|-- prediction.py  # 用于预测的代码
|-- requirements.txt # 依赖文件
|-- script.sh # 便于运行的脚本文件
|-- test_without_label.txt # 本次实验的提交答案
|-- train.py # 用于训练的模型的代码
```

预测文件为test_without_label.txt！！！！

## 执行代码的流程

代码参数包括：
```
--model 用于选择模型，可选参数有cat, add, CLMLF
--epoch 用于设置训练轮数
--warmup 用于设置预热步数 
--weight_decay 用于设置学习率衰减
--lr 用于设置学习率
--train 用于设置训练或是预测，可选参数有train, test
--fusion 用于设置是否为消融实验，可选参数有all, text, image（all代表不做消融，text代表只有文本，image代表只有图片）
```

运行方式1：在终端运行
```
python main.py --model cat --epoch 10 --warmup 20 --weight_decay 0.01 --lr 0.000004 --train train --fusion all
```

运行方式2：在终端运行
```
chmod +x script.sh
./script.sh
```

终端显示内容为每一轮的训练与验证的损失，准确率，精确度，召回率，F1值

## 参考

[1]  Zhen Li, Bing Xu, Conghui Zhu, and Tiejun Zhao. CLMLF:a contrastive learning and multi-layer fusion method for multimodal sentiment detection. In Findings of the Association for Computational Linguistics: NAACL 2022. Association for Com- putational Linguistics, 2022.

[2] Chuhan Wu, Fangzhao Wu, Tao Qi, Yongfeng Huang, and Xing Xie. Fastformer: Additive attention can be all you need. CoRR, abs/2108.09084, 2021.

[3] Quoc-Tuan Truong, Hady W. Lauw. VistaNet: Visual Aspect Attention Network for Multimodal Sentiment Analysis

[4] https://github.com/Link-Li/CLMLF

[5] https://github.com/liyunfan1223/multimodal-sentiment-analysis

