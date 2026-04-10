import os
import sys
import argparse
import logging
import warnings
import random
import numpy as np
import torch
from tqdm import tqdm

import utils
import datasets
import test_BLIP2 as test
from data_utils import squarepad_transform, targetpad_transform
import setproctitle
from lavis.models import load_model_and_preprocess

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
proc_title = "test-only"
setproctitle.setproctitle(proc_title)

# ===================== 测试参数 =====================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cirr', help="数据集类型")
parser.add_argument('--fashioniq_path', default="/root/autodl-tmp/data/CIR_data/fashion_iq_data/")
parser.add_argument('--cirr_path', default="/root/autodl-tmp/data/CIR_data/cirr_data/CIRR/")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--model_dir', default='/root/autodl-tmp/qgz/MELT/checkpoint/', help='模型所在文件夹')
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()

# ===================== 固定配置 =====================
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

# ===================== 加载数据集 =====================
def load_test_dataset():
    transform = "targetpad"
    input_dim = 224
    target_ratio = 1.25
    preprocess = targetpad_transform(target_ratio, input_dim)

    if args.dataset == 'fashioniq':
        dataset = datasets.FashionIQ(path=args.fashioniq_path, transform=preprocess)
    elif args.dataset == 'cirr':
        dataset = datasets.CIRR(path=args.cirr_path, transform=preprocess, case_look=False)
    else:
        raise Exception("不支持的数据集")
    return dataset

# ===================== 加载模型 =====================
def load_trained_model():
    model_path = os.path.join(args.model_dir, "best_model.pt")
    print(f"✅ 加载训练好的模型: {model_path}")
    model = torch.load(model_path, map_location=args.device)
    model.eval()
    model = model.to(args.device)

    # 加载文本处理器
    _, _, txt_processors = load_model_and_preprocess(
        name="Blip2QformerCir",
        model_type="pretrain",
        is_eval=True,
        device=args.device
    )
    return model, txt_processors

# ===================== 执行测试 =====================
if __name__ == "__main__":
    # print("===== 只测试模式 | 无训练 =====")
    dataset = load_test_dataset()
    model, txt_processors = load_trained_model()

    with torch.no_grad():
        if args.dataset == "fashioniq":
            for cate in ["dress", "shirt", "toptee"]:
                print(f"\n正在测试: {cate}")
                res = test.test(args, model, dataset, cate, txt_processors)
                print(f"结果 {cate}: {res}")

        elif args.dataset == "cirr":
            print("\n正在测试 CIRR 验证集")
            res = test.test_cirr_valset(args, model, dataset, txt_processors)
            print(f"结果: {res}")

    print("\n🎉 测试全部完成！")