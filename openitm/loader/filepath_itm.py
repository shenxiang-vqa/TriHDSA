class Path:
    def __init__(self):
        self.DATASET_ROOT_PATH = '/home/shenxiang/sda/Datasets/ITM/data/itm/'
        self.IMGFEAT_ROOT_PATH = '/home/shenxiang/sda/Datasets/ITM/data/itm/'

        # BERT预训练模型路径
        #self.PRETRAINED_PATH = '/home/xuyangshuyi/Openvqa_xy/pretrained_models/roberta-base-eng/'
        # self.PRETRAINED_PATH = '/home/xuyangshuyi/Openvqa_xy/pretrained_models/bert-base-uncased/'
        # self.PRETRAINED_PATH = '/home/xuyangshuyi/Openvqa_xy/pretrained_models/albert-base-v2-uncased/'

        self.CKPT_PATH = './logs/ckpts/'

        self.IMGFEAT_PATH = {
            'coco': {
                'train': self.IMGFEAT_ROOT_PATH + 'bua-r101-fix36/train2014/',
                'val': self.IMGFEAT_ROOT_PATH + 'bua-r101-fix36/val2014/',
            },
            'flickr': {
                'train': self.IMGFEAT_ROOT_PATH + 'flickr_bua-r101-fix36/',
            },
        }

        self.CAPTION_PATH = {
            'coco': {
                'train-caps': self.DATASET_ROOT_PATH + 'coco_precomp/train_caps.txt',
                'train-ids': self.DATASET_ROOT_PATH + 'coco_precomp/train_ids.txt',
                'dev-caps': self.DATASET_ROOT_PATH + 'coco_precomp/dev_caps.txt',
                'dev-ids': self.DATASET_ROOT_PATH + 'coco_precomp/dev_ids.txt',
                'test-caps': self.DATASET_ROOT_PATH + 'coco_precomp/test_caps.txt',
                'test-ids': self.DATASET_ROOT_PATH + 'coco_precomp/test_ids.txt',
                'testall-caps': self.DATASET_ROOT_PATH + 'coco_precomp/testall_caps.txt',
                'testall-ids': self.DATASET_ROOT_PATH + 'coco_precomp/testall_ids.txt',
            },
            'flickr': {
                'train-caps': self.DATASET_ROOT_PATH + 'f30k_precomp/train_caps.txt',
                'train-ids': self.DATASET_ROOT_PATH + 'f30k_precomp/train_ids.txt',
                'dev-caps': self.DATASET_ROOT_PATH + 'f30k_precomp/dev_caps.txt',
                'dev-ids': self.DATASET_ROOT_PATH + 'f30k_precomp/dev_ids.txt',
                'test-caps': self.DATASET_ROOT_PATH + 'f30k_precomp/test_caps.txt',
                'test-ids': self.DATASET_ROOT_PATH + 'f30k_precomp/test_ids.txt',
                'orin': self.DATASET_ROOT_PATH + 'dataset_flickr30k.json',
            },
        }

        self.EVAL_PATH = {
            'result_test': self.CKPT_PATH + 'result_test/',
            'tmp': self.CKPT_PATH + 'tmp/',
            'arch': 'arch/',
        }
