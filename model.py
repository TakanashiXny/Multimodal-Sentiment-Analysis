import torch.nn.modules as nn
import torchvision.models as cv_models
import torch
from transformers import BertConfig, BertModel, BertPreTrainedModel, BertLayer, ResNetModel
from transformers import AutoModel, ResNetModel
from transformers.models.roberta.modeling_roberta import RobertaEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_extended_attention_mask(attention_mask):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class FuseModel(nn.Module):
    '''论文中的MLF Module
    
    Attributes:
        config: 模型的配置
        text_model: 预训练文本模型
        image_model: 预训练图片模型
        train_dim: 训练维度
        hidden_dim: 隐藏层维度
    '''
    def __init__(self, config) -> None:
        super(FuseModel, self).__init__()
        self.config = config
        self.text_model = BertModel.from_pretrained('./model/bert-base-uncased')
        self.image_model = ResNetModel.from_pretrained('./model/resnet50')
        self.train_dim = config['train_dim']
        self.hidden_dim = 768

        self.text_change = nn.Sequential(
            nn.Linear(self.hidden_dim, self.train_dim),
            nn.GELU()
        )

        self.image_change = nn.Sequential(
            nn.Linear(2048, self.train_dim),
            nn.Tanh()
        )

        self.output_attention = nn.Sequential(
                nn.Linear(self.train_dim, self.train_dim // 2),
                nn.GELU(),
                nn.Linear(self.train_dim // 2, 1)
            )

        self.bert_config = BertConfig('./model/bert-base-uncased')
        self.TransformerEncoder = RobertaEncoder(config=self.bert_config)

        self.device = config['device']

    def forward(self, text, image):
        
        # 处理文本
        text_features = self.text_model(**text)
        text_features = text_features.last_hidden_state
        text_features = self.text_change(text_features)

        # 处理图片
        image_features = self.image_model(image)
        image_features = image_features.last_hidden_state.view(-1, 49, 2048).contiguous()
        image_features = self.image_change(image_features)

        image_attenion = torch.ones((text.attention_mask.size(0), 49)).to(self.device)
        extended_attention_mask = get_extended_attention_mask(image_attenion)
        # 此处形状出现过bug
        image_encoded = self.TransformerEncoder(image_features,
                                                attention_mask=None,
                                                head_mask=None,
                                                encoder_hidden_states=None,
                                                encoder_attention_mask=extended_attention_mask,
                                                past_key_values=None,
                                                output_attentions=self.bert_config.output_attentions,
                                                output_hidden_states=self.bert_config.output_hidden_states,
                                                return_dict= self.bert_config.use_return_dict
                                                ).last_hidden_state  

        # 合并文字和图片
        text_image_features = torch.cat((text_features, image_encoded), dim=1)  ### [batch_size, 64+49, 768]

        text_image_attention = torch.cat((image_attenion, text.attention_mask), dim=1)
        extended_attention_mask = get_extended_attention_mask(text_image_attention)
        # text_image_state_encoded = self.TransformerEncoder(text_image_hidden_state)
        text_image_encoded = self.TransformerEncoder(text_image_features,
                                                    attention_mask=extended_attention_mask,
                                                    encoder_hidden_states=None,
                                                    encoder_attention_mask=extended_attention_mask,
                                                    past_key_values=None,
                                                    output_attentions=self.bert_config.output_attentions,
                                                    output_hidden_states=self.bert_config.output_hidden_states,
                                                    return_dict= self.bert_config.use_return_dict
                                                    ).last_hidden_state

        text_image_output = text_image_encoded.contiguous()

        text_image_mask = text_image_attention.permute(1, 0).contiguous()
        text_image_mask = text_image_mask[0:text_image_output.size(1)]
        text_image_mask = text_image_mask.permute(1, 0).contiguous()

        text_image_alpha = self.output_attention(text_image_output)
        text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_image_mask == 0, -1e9)
        text_image_alpha = torch.softmax(text_image_alpha, dim=-1)

        text_image_output = (text_image_alpha.unsqueeze(-1) * text_image_output).sum(dim=1)

        return text_image_output
    

class CLModel(nn.Module):
    '''论文中的CLMLF模型
    
    Attributes:
        fuse_model: 融合模型
        temperature: 模型参数变化
        device: 模型运行的设备
        train_dim: 训练维度
        dropout: 丢弃概率
        linear_change: 线性层
        classifier: 分类层
    '''
    def __init__(self, config) -> None:
        super(CLModel, self).__init__()
        self.fuse_model = FuseModel(config)
        self.temperature = config['temperature']
        self.device = config['device']
        self.train_dim = config['train_dim']
        self.dropout = config['dropout']
        self.linear_change = nn.Sequential(
            nn.Linear(self.train_dim, self.train_dim),
            nn.GELU(),
            nn.Linear(self.train_dim, self.train_dim)

        )

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.train_dim, self.train_dim // 2),
            nn.GELU(),
            nn.Linear(self.train_dim // 2, 3)
        )

    def forward(self, text, image, augment_text, augment_image, labels, target_labels):
        text_image = self.fuse_model(text, image)
        output = self.classifier(text_image)

        if augment_text != None:
            aug_text_image_features = self.fuse_model(augment_text, augment_image)
            org_res_change = self.linear_change(text_image)
            aug_res_change = self.linear_change(aug_text_image_features)

            l_pos_neg = torch.einsum('nc,ck->nk', [org_res_change, aug_res_change.T])
            cl_lables = torch.arange(l_pos_neg.size(0))
            cl_lables = cl_lables.to(self.device)

            l_pos_neg /= self.temperature

            l_pos_neg_self = torch.einsum('nc,ck->nk', [org_res_change, org_res_change.T])
            l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1)
            l_pos_neg_self = l_pos_neg_self.view(-1)

            cl_self_labels = target_labels[labels[0]]
            for index in range(1, text_image.size(0)):
                cl_self_labels = torch.cat((cl_self_labels, target_labels[labels[index]] + index*labels.size(0)), 0)

            l_pos_neg_self = l_pos_neg_self / self.temperature
            cl_self_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
            cl_self_loss = - cl_self_loss.sum() / cl_self_labels.size(0)

            return output, l_pos_neg, cl_lables, cl_self_loss
        else:
            return output
        

class myModel(nn.Module):
    '''拼接模型
    
    Attributes:
        text_encoder: 文本预训练模型
        image_encoder: 图片预训练模型
        num_labels: 标签数量
    '''
    def __init__(self, num_labels=3, 
                 text_encoder='./model/bert-base-uncased/', 
                 image_encoder='./model/resnet50/') -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        self.image_encoder = ResNetModel.from_pretrained(image_encoder)
        self.num_labels = num_labels
        self.W = nn.Linear(
            in_features = self.text_encoder.config.hidden_size + 2048,
            out_features = num_labels
        )

        self.text_change = nn.Linear(self.text_encoder.config.hidden_size, num_labels)
        self.image_change = nn.Linear(2048, num_labels)

        self.flatten = nn.Flatten()
        self.linear  = nn.Linear(49152, 768)

        self.text_classifier = nn.Linear(self.text_encoder.config.hidden_size, num_labels)
        self.image_classifier= nn.Linear(2048, num_labels)

    def forward(self, text, image):

        if text is not None and image is not None:
            # 先对数据进行编码
            text_feature = self.text_encoder(**text)
            image_feature = self.image_encoder(image)

            # 对编码数据进行信息提取
            text_feature = text_feature.last_hidden_state[:, 0, :]
            image_feature = image_feature.last_hidden_state.view(-1, 49, 2048)
            image_feature = image_feature.max(1)[0]

            # 将文本数据和图片数据拼接在一起
            features = torch.cat((text_feature, image_feature), 1)

            logits = self.W(features)

            return logits
        
        elif text is not None:
            text_output = self.text_encoder(**text)
            text_feature = text_output.last_hidden_state
            # print(text_feature.shape)
            
            logits = self.text_classifier(self.linear(self.flatten(text_feature)))
            return logits
        
        else:
            img_feature = self.visual_encoder(image).last_hidden_state.view(-1, 49, 2048).max(1)[0]
            logits = self.image_classifier(img_feature)
            
            return logits
        
        
class myModel2(nn.Module):
    '''加法模型
    
    Attributes:
        text_encoder: 文本预训练模型
        image_encoder: 图片预训练模型
        num_labels: 标签数量
    '''
    def __init__(self, num_labels=3, 
                 text_encoder='./model/bert-base-uncased/', 
                 image_encoder='./model/resnet50/') -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        self.image_encoder = ResNetModel.from_pretrained(image_encoder)
        self.num_labels = num_labels

        self.image_align = nn.Linear(in_features=2048, out_features=768)
        self.image_align2 = nn.Linear(in_features=49 , out_features=64)

        self.tanh = nn.Tanh()
        self.W = nn.Linear(in_features=768, out_features=768)

        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(in_features=768*64, out_features=num_labels)

        self.w = nn.Linear(768, 1)
        self.soft = nn.Softmax(dim=0)


    def forward(self, text, image):
        feature = 0

        if text is not None and image is not None:
            
            # 先对数据进行编码
            text_feature = self.text_encoder(**text)
            image_feature = self.image_encoder(image)

            # 对编码数据进行信息提取
            text_feature = text_feature.last_hidden_state  # 大小为[batch_size, 64, 768]
            image_feature = image_feature.last_hidden_state.view(-1, 49, 2048)
            
            # 将图片数据大小与文本对齐
            image_align = self.image_align(image_feature) # 大小为[batch_size, 49, 768]
                                                          # 第二列不相等
            
            tmp = self.image_align2(image_align.permute(0, 2, 1)).permute(0, 2, 1)
            image_align = tmp

            # 为每一个样本创建一个向量
            size = text_feature.shape[0]
            for i in range(size):
                alpha_text = self.soft(self.w(text_feature[i]))
                alpha_image = self.soft(self.w(image_align[i]))
                tmp_feature = alpha_text * text_feature[i] + alpha_image * image_align[i]

                tmp_feature = self.W(self.tanh(tmp_feature))

                if i == 0:
                    feature = tmp_feature.unsqueeze(0)
                else:
                    feature = torch.cat((feature, tmp_feature.unsqueeze(0)), 0)

            out = self.classifier(self.flatten(feature))

            return out
        
        elif text is None:
            image_feature = self.image_encoder(image)
            image_feature = image_feature.last_hidden_state.view(-1, 49, 2048)
            image_align = self.image_align(image_feature) # 大小为[batch_size, 49, 768]
                                                          # 第二列不相等
            
            tmp = self.image_align2(image_align.permute(0, 2, 1)).permute(0, 2, 1)
            image_align = tmp

            size = image_feature.shape[0]
            for i in range(size):
                tmp_feature = image_align[i]

                tmp_feature = self.W(self.tanh(tmp_feature))

                if i == 0:
                    feature = tmp_feature.unsqueeze(0)
                else:
                    feature = torch.cat((feature, tmp_feature.unsqueeze(0)), 0)

            out = self.classifier(self.flatten(feature))

            return out

        else:
            text_feature = self.text_encoder(**text)
            text_feature = text_feature.last_hidden_state  # 大小为[batch_size, 128, 768]

            size = text_feature.shape[0]
            for i in range(size):
                tmp_feature = text_feature[i]

                tmp_feature = self.W(self.tanh(tmp_feature))

                if i == 0:
                    feature = tmp_feature.unsqueeze(0)
                else:
                    feature = torch.cat((feature, tmp_feature.unsqueeze(0)), 0)

            out = self.classifier(self.flatten(feature))

            return out