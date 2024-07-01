import argparse
import itertools
import json
import os
import re
from functools import partial

import torch
from torchvision.ops.boxes import box_area
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist

import sys

sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])
from data.bbox_util import Expand2square
# from util.quant import quantize
from fairscale.nn.model_parallel import initialize as fs_init
from model.meta import MetaModel
from util.tensor_parallel import load_tensor_parallel_model_list
from util.misc import init_distributed_mode
from data.conversation.lib import conv_templates, SeparatorStyle
from data.transform import get_transform


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def collate_fn(batches):
    texts = [_['text'] for _ in batches]
    
    hws = [_['hw'] for _ in batches]
    filenames = [_['filename'] for _ in batches]
    classes = [_['class'] for _ in batches]
    input_image = torch.cat([_['image'] for _ in batches])

    # input_ids = tokenizer.encode(texts, return_tensors='pt', padding='longest')

    return classes, filenames, texts, input_image, hws


from PIL import Image

import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class PadToSquare:
    def __init__(self, background_color):
        """
        pad an image to squre (borrowed from LLAVA, thx)
        :param background_color: rgb values for padded pixels, normalized to [0, 1]
        """
        self.bg_color = tuple(int(x*255) for x in background_color)

    def __call__(self, img: Image.Image):
        width, height = img.size
        if width == height:
            return img
        elif width > height:
            result = Image.new(img.mode, (width, width), self.bg_color)
            result.paste(img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(img.mode, (height, height), self.bg_color)
            result.paste(img, ((height - width) // 2, 0))
            return result

def T_padded_resize(size=224):
    t = transforms.Compose([
        PadToSquare(background_color=(0.48145466, 0.4578275, 0.40821073)),
        transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return t

class ESDDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, prompt, image_size=224):
        # self.meta_l = json.load(data_path)
        with open(data_path) as f:
            self.meta_l = json.load(f)
        # self.tokenizer = tokenizer
        self.prompt = prompt
        self.transform_val = get_transform('only_norm', image_size)
        

    def __len__(self):
        return len(self.meta_l)

    def __getitem__(self, idx):
        # print('data_idx: ',self.datas[idx])
        data = self.meta_l[idx]
        
        filename = data['image'].replace('D:/23-24RA/LLM_ROBOT/ESD_video/high_light_images','data/pad_images')
        text = data['conversations'][0]["value"]

        image = Image.open(filename).convert('RGB')
        w, h = image.width, image.height

        
        image = self.transform_val(image).unsqueeze(0)
        
        question = self.prompt.format(text)



        return {
            'class':text,
            'filename':filename,
            'text': question,
            'image': image,
            'hw': (h, w),
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

    def get_args_parser():
        parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
        # Model parameters
        parser.add_argument('--llama_type', default='llama_qformerv2', type=str, metavar='MODEL',
                            help='type of llama')
        parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                            help='Path to llama model config')
        parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                            help='path to tokenizer.model')
        parser.add_argument('--img_root', type=str, default="./data/nocaps/images",
                            help='path to tokenizer.model')
        parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                            help='directory containing pre-trained checkpoints')

        parser.add_argument('--device', default='cuda',
                            help='device for inference')
        parser.add_argument('--model_parallel_size', default=1, type=int)

        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--batch_size', default=8, type=int)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--dist_on_itp', action='store_true')
        parser.add_argument('--dist_url', default='env://',
                            help='url used to set up distributed training')
        parser.add_argument('--quant', action="store_true", default=False,
                            help="enable quantization")
        parser.add_argument("--input_size", type=int, default=224)
        parser.add_argument('--dataset', default='nocaps', type=str)
        parser.add_argument("--max_seq_length", type=int, default=2048)
        parser.add_argument('--savename', type=str, default="coco_raw",
                            help='path to tokenizer.model')

        return parser


    args = get_args_parser().parse_args()

    # define the model
    init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=True, max_seq_len=args.max_seq_length)
    print(f"load pretrained from {args.pretrained_path}")
    load_result = load_tensor_parallel_model_list(model, args.pretrained_path)
    print("load result: ", load_result)
    # print("Quantizing model to 4bit!")

    # from transformers.utils.quantization_config import BitsAndBytesConfig

    # quantization_config = BitsAndBytesConfig.from_dict(
    #     config_dict={
    #         "load_in_8bit": False,
    #         "load_in_4bit": True,
    #         "bnb_4bit_quant_type": "nf4",
    #     },
    #     return_unused_kwargs=False,
    # )
    # quantize(model, quantization_config)

    # print("Model = %s" % str(model))
    model.bfloat16().cuda()

    prompt = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.

###Human: {} Only output the motion of Forceps.
###Assistant:"""
    # prompt = '''Can you point out {} in the image and provide the coordinates of its location? The answer is: ['''
    dataset = ESDDataset( data_path='data/annotations/smllm_test.json' ,
                             prompt=prompt, image_size=getattr(model.llma, 'image_size', 224))
    print('data lens', len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn),
    )

    max_gen_len = 200
    gen_t = 0.0
    top_p = 0.75
    all_answers =[]
    all_gt_box=[]
    all_hws=[]
    all_classes =[]
    all_filenames=[]
    det_loss = torch.nn.L1Loss()
    if dist.get_rank() == 0:
        global tqdm
    else:
        tqdm = lambda x: x  
    idx = 0
    for classes, filenames, _prompt, image, hws in tqdm(dataloader):
        image = image.cuda(non_blocking=True)
        # if idx==10:
        #     break
        if fs_init.get_model_parallel_world_size() > 1:
            dist.broadcast_object_list([_prompt, image, max_gen_len, gen_t, top_p])
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            results = model.generate(_prompt, image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
            print('prompt:', _prompt)
            print(f'pred: {results[0]}')
            truncated = [] 
            for hw, answer in zip( hws, results):
                answer = answer.split('###')[0]
                truncated.append(answer)
            
        if fs_init.get_data_parallel_world_size() > 1:
            truncated_allgather = [None for _ in range(fs_init.get_data_parallel_world_size())]
            hw_allgather = [None for _ in range(fs_init.get_data_parallel_world_size())]
            class_allgather = [None for _ in range(fs_init.get_data_parallel_world_size())]
            filename_allgather = [None for _ in range(fs_init.get_data_parallel_world_size())]
            dist.all_gather_object(truncated_allgather, truncated, fs_init.get_data_parallel_group())
    
            dist.all_gather_object(hw_allgather, hws, fs_init.get_data_parallel_group())
            dist.all_gather_object(class_allgather, classes, fs_init.get_data_parallel_group())
            dist.all_gather_object(filename_allgather, filenames, fs_init.get_data_parallel_group())
            
            truncated = list(sum(zip(*truncated_allgather), ()))
            hws = list(sum(zip(*hw_allgather), ()))
            classes = list(sum(zip(*class_allgather), ()))
            filenames = list(sum(zip(*filename_allgather), ()))
            print('truncated_allgather',len( truncated_allgather), len(truncated), len(hws), len(classes), len(filenames))
        
        if dist.get_rank() == 0:
            all_answers.extend(truncated) 
            
            all_hws.extend(hws) 
            all_classes.extend(classes)
            all_filenames.extend(filenames) 
            print('extended truncated_allgather',len( all_answers), len(all_hws), len(all_classes), len(all_filenames))
            
        idx = idx+1      
       

    print('all together:', len(all_answers))
    
    
    if torch.distributed.get_rank() == 0:
        # correct = total_cnt = 0
        with open(os.path.join('evaluation',args.savename+'.json'),'w') as wf:
            for answer, filename in zip(all_answers, all_filenames):
                print('output', answer)
            # print(output['answer'])
            

                final_ans = {'pred': answer, 'filename': filename}
                wf.write(json.dumps(final_ans) + '\n')

            print("write {} answers to {}".format(len(all_answers), args.savename))
    torch.distributed.barrier()
    
