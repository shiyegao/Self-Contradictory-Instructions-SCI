from .information_conflict import Rule_violation_conflict,Object_attribute_conflict
from .instruction_conflict import Forbidden_word_conflict,Exclusive_instruction_conflict,Generate_story_agent
import openai
import yaml
import argparse
import json
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description='Script for 4 tasks')
parser.add_argument('--task',type=int,choices=[1,2,3,4],default=1,
                    help='''Four types of L-L conflict:
1.Rule conflict
2.Attribute conflict
3.Exclusion conflict
4.Forbidden conflict''')
parser.add_argument('--total_conflicts',type=int,default=2500,help='number of total conflicts in dataset')
parser.add_argument('--config',type=str,default='openai.yaml',help='config for OPENAI')
parser.add_argument('--dataset_path',type=str,default='../dataset',help='path to save dataset')

parser.add_argument('--num_attribute',type=int,default=8,help='argument for task 2')
parser.add_argument('--num_elements',type=int,default=3,help='argument for task 3')
parser.add_argument('--story_seed_path',type=str,default='seeds.json',help='argument for task 3')
args = parser.parse_args()



OPENAI_CONFIG = yaml.load(open(args.config), Loader=yaml.FullLoader)
PROMPT_TEMPLATE = yaml.load(open('LL_prompt.yaml'),Loader=yaml.FullLoader)
openai.api_key = OPENAI_CONFIG['OPENAI_KEY']
try:
    openai.base_url = OPENAI_CONFIG['OPENAI_API_BASE']
except:
    pass
model = OPENAI_CONFIG['MODEL_NAME']
max_retry = OPENAI_CONFIG['MAX_RETRY']




os.makedirs(args.dataset_path,exist_ok=True)

if args.task == 1:
    print('Task 1:\nRule conflict')
    LLM = Rule_violation_conflict(model = model,prompt_template=PROMPT_TEMPLATE,max_retry=max_retry)
    rounds = int(args.total_conflicts/20)  #LLM generate 20 conflicts each round
    target_path = os.path.join(args.dataset_path, 'Rule_conflict.jsonl')
    for i in tqdm(range(rounds),desc='Rule_conflict'):
        LLM.generate_triple_in_batch(dataset_path=target_path)
        
elif args.task == 2:
    print('Task 2:\nAttribute conflict')
    LLM = Object_attribute_conflict(model=model,prompt_template=PROMPT_TEMPLATE,max_retry=max_retry,num_attribute=args.num_attribute)
    target_path = os.path.join(args.dataset_path, 'Attribute_conflict.jsonl')
    LLM.generate_conflict(N=int(args.total_conflicts/args.num_attribute),dataset_path=target_path)
    
elif args.task == 3:
    print('Task 3:\nExclusion conflict')
    LLM = Generate_story_agent(model=model,prompt_template=PROMPT_TEMPLATE,max_retry=max_retry)
    EIC = Exclusive_instruction_conflict()
    num_stories = int(args.total_conflicts/EIC.count_total_conflict_pair())
    for i in tqdm(range(num_stories),desc='genrating stories'):
        LLM.generate_storyline(num=args.num_elements,seed_path=args.story_seed_path)
    # save the stories
    with open("newstory.jsonl","w") as f:
        json.dump(LLM.stories,f)
        
    target_path = os.path.join(args.dataset_path, 'Exclusion_conflict.jsonl')
    for story in tqdm(LLM.stories,desc='concat[inst1, story, inst2]'):
        EIC.generate_conflict(para=story,path=target_path)
    EIC.generate_neglect_conflict(N=args.total_conflicts,story_path='newstory.jsonl',dataset_path=target_path)    

        
        
elif args.task == 4:
    print('Task 4:\nForbidden conflict')
    LLM = Forbidden_word_conflict(model=model,prompt_template=PROMPT_TEMPLATE,max_retry=max_retry)
    ## The list can be adjusted to need; number is hueristically referenced "high-quality" limit, can be adjusted
    #generate_list = [("historic event","object",75),("muscial instrument","object",75),("famous architechture","object",100),("classic books","object",150),("famous people","object",250),("medicine","object",75),("large company","object",200),("classic movie","object",100),("nations","object",100),("city","object",100),("common animal","object",100),("common plant","object",100),("classic artwork","object",100),("digital game","object",100),("philosophy","concept",75),("mathematics","concept",75),("chemistry","concept",75),("biology","concept",75),("physics","concept",75),("economics","concept",75),("geology","concept",75),("psychology","concept",75)]
    ## 2250 Raw data; After clean with cleaner agent, this number should drop a bit to about ~2000 level?
    
    # new list for generation #2800
    new_list = [
                ("Ancient Civilization", "object", 50),
                ("Musical Genre", "object", 100),
                ("Inventor", "object", 100),
                ("Scientific Theory", "concept", 100),
                ("Astronomical Phenomenon", "concept", 100),
                ("Literary Genre", "concept", 100),
                ("Computer Programming Language", "concept", 50),
                ("Era in History", "concept", 100),
                ("famous landmark", "concept", 100),
                ("Natural Disasters", "concept", 100),
                ("Scientific Experiments", "concept", 100),



                ("Ancient Philosophers", "object", 100),
                ("Music Bands", "object", 100),
                ("Musical Instruments", "concept", 100),
                ("Ancient Ruins", "object", 100),
                ("Famous Poets", "object", 100),
                ("Natural Wonders", "object", 100),
                ("Palaeobios","object",100),
                ("Ancient Empires", "concept", 100),
                ("Scientific Experiments", "concept", 100),
                ("Famous Authors", "object", 100),


                ("Economic terms","object",100),
                ("mathematical theorem","object",100),
                ("Laws of physics", "object", 100),
                ("Famous Physicians", "object", 100),
                ("Laws of Chemistry", "object", 100),
                ("Financial Concepts", "concept", 100),
                ("Famous Playwrights", "object", 100),
                ("Notable Biologists", "object", 100),
                ]
    
    target_path = os.path.join(args.dataset_path, 'Forbidden_conflict.jsonl')
    for category,object,num in tqdm(new_list,desc="Total"):
        LLM.sample_and_create(category = category,object=object,top=num,path =args.dataset_path + target_path)