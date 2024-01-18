from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from PIL import Image
import torchvision.transforms as transforms


class myDataset(Dataset):
    '''
    构造模型可用的数据类型
    
    Attributes:
        config: 模型基本配置参数
        df: 整理后的数据表
        data_type: 0代表训练集, 1代表验证集, 2代表测试集
        text_image: 0代表包含文本和图片, 1代表只需要文本, 2代表只需要图片
        image_transform: 裁剪图片的方法
        back_translation: 是否需要增强数据
    '''

    def __init__(self, df, data_type, text_image, image_transform, back_translation) -> None:
        super().__init__()
        self.df = df
        self.data_type = data_type
        self.text_image = text_image
        self.image_transform = image_transform
        self.back_translation = back_translation

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        if self.text_image != 2:
            # 取出文字内容
            text = self.df.iloc[index]['texts']
            if self.back_translation:
                text_aug = self.df.iloc[index]['text_aug']
        if self.text_image != 1:
            # 取出图片内容
            image_id = self.df.iloc[index]['id']
            image_path = './data/images/' + str(image_id) + '.jpg'
            image = Image.open(image_path)
            img = self.image_transform(image)
        # 取出标签
        if self.data_type != 2:
            emotion_label = self.df.iloc[index]['emotion_label']

        if self.back_translation:
            if self.text_image == 0:
                return text, text_aug, emotion_label, img
            elif self.text_image == 1:
                return text, text_aug, emotion_label, 
            elif self.text_image == 2:
                return emotion_label, img
        else:
            if self.text_image == 0:
                if self.data_type != 2:
                    return text, emotion_label, img
                else:
                    return text, img
            elif self.text_image == 1:
                if self.data_type != 2:
                    return text, emotion_label
                else:
                    return text
            elif self.text_image == 2:
                if self.data_type != 2:
                    return emotion_label, img
                else:
                    return img
        

def makeDataLoader(df, data_type, batch_size, back_translation):
    '''
    构造DataLoader

    Args:
        df: 需要使用的DataFrame数据
        data_type: 0代表训练集, 1代表验证集, 2代表测试集
        batch_size: 批大小
        back_translation: 是否需要强化数据
    '''
    
    # ResNet-50 settings
    img_size = 224
    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

    if data_type == 0:
        train_transform_func = transforms.Compose(
                [transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]
        )
        dataset = myDataset(df, 0, 0, train_transform_func, back_translation)
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, batch_size, sampler=sampler)
    elif data_type == 1:
        eval_transform_func = transforms.Compose(
                [transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ]
        )
        dataset = myDataset(df, 1, 0, eval_transform_func, back_translation)
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, batch_size, sampler=sampler)
    elif data_type == 2:
        eval_transform_func = transforms.Compose(
                [transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ]
        )
        dataset = myDataset(df, 2, 0, eval_transform_func, back_translation)
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, batch_size, shuffle=False)

    return data_loader
    

