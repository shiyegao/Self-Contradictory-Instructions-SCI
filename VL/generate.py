from PIL import Image, ImageDraw, ImageFont
import yaml
import numpy as np,tqdm
from visionlanguage import OCRConflictGenerator, FigureConflictGenerator, GeometricConflictGenerator, SemanticConflictGenerator
import openai,os,re
import argparse


parser = argparse.ArgumentParser(description='Script for 4 tasks')
parser.add_argument('--task',type=int,choices=[1,2,3,4],default=1,
                    help='''Four types of V-L conflict:
1.OCR conflict
2.Figure conflict
3.Geometric conflict
4.Semantic conflict''')
parser.add_argument('--config',type=str,default='openai.yaml',help='config for OPENAI')
parser.add_argument('--total_num',type=int,default=2000,help='total number of generation')
parser.add_argument('--target_dir',type=str,default=None,help='directories for saving generated images')

args = parser.parse_args()

OPENAI_CONFIG = yaml.load(open(args.config), Loader=yaml.FullLoader)
COLOR_CONFIG = yaml.load(open('colors.yaml'),Loader=yaml.FullLoader)
PROMPT_TEMPLATE = yaml.load(open('VL_prompt.yaml'),Loader=yaml.FullLoader)
os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_KEY']
openai.api_key = OPENAI_CONFIG['OPENAI_KEY']
try:
    openai.api_base = OPENAI_CONFIG['OPENAI_API_BASE']
except:
    pass
model = OPENAI_CONFIG['MODEL_NAME']
max_retry = OPENAI_CONFIG['MAX_RETRY']



os.makedirs(args.target_dir,exist_ok=True)

if args.task == 1:
    ocg = OCRConflictGenerator(model=model,prompt_template=PROMPT_TEMPLATE,max_retry=max_retry,img_dir=args.target_dir)
    num_sentnece = int(args.total_num/82)
    ocg.grow_sentence_list(num=num_sentnece)
    ocg.create()

elif args.task == 2:
    fcg = FigureConflictGenerator(model=model,prompt_template=PROMPT_TEMPLATE,max_retry=max_retry)
    fcg.get_dict()
    fcg.create(target_dir=args.target_dir)

elif args.task == 3:
    gcg = GeometricConflictGenerator()
    if args.target_dir:
        gcg.create(num=args.total_num,target_dir=args.target_dir)
        gcg.choose(path=args.target_dir)
    else:
        gcg.create(num=args.total_num)
        gcg.choose()

elif args.task == 4:
    scg = SemanticConflictGenerator(model=model,prompt_template=PROMPT_TEMPLATE,max_retry=max_retry)
    
    if args.target_dir:
        scg.create(target_dir=args.target_dir)
        scg.choose(target_dir=args.target_dir)
    else:
        scg.create()
        scg.choose()

