import json
import random
import re
import itertools as it
from tqdm import tqdm
import time
from utils import LLMObject,list_to_string





class Forbidden_word_conflict(LLMObject):
    """
        Generate don't mention conflicts
    """
    def __init__(self, model, prompt_template, max_retry: int = 10) -> None:
        super().__init__(model, prompt_template, max_retry)
        self.objects = ["animal"]
        self.conflict_list = []
    

    def sample_and_create(self,category:str,object:str="object",path:str = "instruction_conflict.jsonl",top=10):
        """
        This method sample N(=top) objects from given category
        and create a question for each object.
        """
        
        
        #----- sample N(=top) objects from the given category
        message  = self.PROMPT_TEMPLATE['SAMPLE'].format(x = category,num=top,object=object)
        #print(message)
        completion,sample = self.CreateWrapper(message=message)
        print(sample)
        # This regex is stupid, maybe try the code in GenerateAgent.generate_seed()
        regex = r"\d+\.\s*(\d+[,A-Za-z]*\d+\s[A-Za-z]*|[A-Za-z\s'\",’\&.-]+|\d+)"
        match_list = re.findall(regex,sample)
        match_list = [s.rstrip('\n') for s in match_list]
        #print(match_list)
        i=0


        # create a question for each object
        for s in tqdm(match_list,desc='generating question for each object'):
            i+=1
            message = self.PROMPT_TEMPLATE['QUESTION'].format(x = s,category=category)
            completion,question = self.CreateWrapper(message=message)
            for idx in range(self.max_retry):
                try:
                    regex = r"Question:\s*(.*)"
                    m_question = re.search(regex,question).group(1).lstrip()
                    #print(m_question)
                    c_question = self.PROMPT_TEMPLATE["ASK"].format(object = s,question = m_question)
                    conf = {"object":s,"question":m_question,"prompt":c_question,"field":category}
                    with open(path,"a") as f:
                        json.dump(conf,f)
                        f.write('\n')
                    break
                except:

                    if idx == self.max_retry-1:
                        c_question = self.PROMPT_TEMPLATE["ASK"].format(object = s,question = question)
                        conf = {"object":s,"question":question,"prompt":c_question,"field":category}
                        with open(path,"a") as f:
                            json.dump(conf,f)
                            f.write('\n')

                        #print(f"{i}:{question}")
                    continue
                               


class Generate_story_agent(LLMObject):
    """
    Generate stories that contain N elements
    """
    def __init__(self, model,prompt_template, max_retry: int = 10) -> None:
        super().__init__(model, prompt_template,max_retry)
        self.stories = []
        self.conflicts = []
        self.qa = []

    def generate_seed(self,num:int=50,path:str="seeds.json"):
        """
            generate some elements as seeds of the story
        """
        content = self.PROMPT_TEMPLATE['SEED_GENERATE'].format(x=num)
        completion,setting_seed = self.CreateWrapper(message=content)
        lines = setting_seed.split("\n")
        settings = []
        for line in lines:
            if line.startswith(str(len(settings) + 1) + "."):
                line = line.replace(str(len(settings) + 1) + ".", "", 1)
                line = line.strip()
                settings.append(line)
        # print(settings)
        with open(path,"w") as f:
            json.dump(settings,f)

    def generate_storyline(self,num:int=3,seed_path:str="seeds.json"):
        """randomly choose {num} elements, and write a story about them

        Args:
            num (int, optional): number of elements to select. Defaults to 3.
            seed_path (str, optional): file path to element seeds. Defaults to "seeds.json".

        Returns:
            str: a story that contains num elements
        """
        with open(seed_path, "r") as f:
            seeds = json.load(f)   
        content = self.PROMPT_TEMPLATE['STORY_GENERATE'].format(x = list_to_string(random.sample(seeds,num)))
        #print("User:\n"+content)
        completion,storyline = self.CreateWrapper(message=content)
        #print("Assistant:\n"+storyline)
        self.stories.append(storyline)
        return storyline      
    
    
    
class Exclusive_instruction_conflict():
    """
        Generate exclusive instruction conflicts
    """
    def __init__(self):
        self.LLM_ACTION = [["Translate the given paragraph to Chinese","Translate the given paragraph to French","Translate the given paragraph to Spanish","Translate the given paragraph to Portuguese","Translate the given paragraph to Germany","Translate the given paragraph to Japanese"],["Repeat exactly every word in the given paragraph","Paraphrase the given paragraph in a rigid and scientific way","Paraphrase the given paragraph in a poetic way"],["Summarize the given paragraph in a long, wordy, tedious way.","Summarize the given paragraph in one short sentence."],["Replace all 'the' to 'ughh' in the given paragraph","Replace all 'the' to 'zzzz' in the given paragraph","Replace all 'the' to 'emmm' in the given paragraph","Replace all 'the' to 'prrr' in the given paragraph"],["Make a comment in favour of the given paragraph, praising it a lot","Make a comment against the given paragraph, criticizing it a lot","Make an unbiased comment on the given paragraph"],["Rewrite the given paragraph such that primary school student are the intended audience","Rewrite the given paragraph such that undergraduates are the intended audience","Rewrite the given paragraph such that industry professionals are the intended audience"],["Rewrite the given paragraph and set it in an East Asian cultural context","Rewrite the given paragraph and set it in an North Europe cultural context","Rewrite the given paragraph and set it in an North America cultural context","Rewrite the given paragraph and set it in an Africa cultural context","Rewrite the given paragraph and set it in an Russian cultural context"]]
        self.instruction_set = [inst for inst_subset in self.LLM_ACTION for inst in inst_subset]
        #LLM_ACTION contains seven types of instructions, any two instruction in the same type are exclusive
        self.total_conflict_pair = self.count_total_conflict_pair()
        print(f'{self.total_conflict_pair} exclusive instruction pairs in total')
        
    def count_total_conflict_pair(self):
        """count the total number of conflict pairs"""
        cnt = 0
        for types in self.LLM_ACTION:
            cnt += len(list(it.permutations(range(len(types)),2)))
        return cnt 
    
    def generate_conflict(self,para:str,path:str='exclusive_instruction_conflict.jsonl'):
        """ concat[ inst 1, story, inst 2 ] to generate conflict dataset

        Args:
            para (str): a paragraph of story
            path (str, optional): file path for saved result. Defaults to 'exclusive_instruction_conflict.jsonl'.
        """
        
        
        for act in self.LLM_ACTION:
            # for any two exclusive instructions, concat with para
            for i,j in it.permutations(range(len(act)),2):
                # This can also be combination instead of permutation

                first_inst = act[i]
                second_inst = act[j]          
                message = f"{first_inst}.\n\n{para}\n\n{second_inst}."
                d = {"content":message,"inst1":first_inst,"inst2":second_inst}

                with open(path,"a") as f:
                    json.dump(d,f)
                    f.write("\n")   


    def generate_neglect_conflict(self,N:int=500,story_path:str='newstory.jsonl',dataset_path:str='neglect_conflict.jsonl'):
        with open(story_path,"r") as stories:
            stories = json.load(stories)
            
        dataset =  open(dataset_path,'a')
        NEGELECT_CONFLICT='Please neglect the next sentence and {inst1}.\nPlease ignore the last sentence and {inst2}\n\n{story}'  
        neglect_pairs = it.permutations(self.instruction_set,2)
        neglect_pairs=list(neglect_pairs)
        print(len(neglect_pairs),'pairs in total')
        rounds = int(N/len(neglect_pairs))
        for i in range(rounds):
            for inst1,inst2 in tqdm(list(neglect_pairs)):
                conflict = NEGELECT_CONFLICT.format(inst1=inst1[0].lower()+inst1[1:],inst2=inst2[0].lower()+inst2[1:],story=random.choice(stories))   
                d = {"content":conflict,"inst1":inst1,"inst2":inst2}
                json.dump(d,dataset)    
                dataset.write("\n")
            
                            
class Cleaner_agent(LLMObject):
    """Cleaner for Forbidden_word_conflict: \n
        check if the question has only one answer——the forbidden word
    """
    def __init__(self, model, prompt_template, max_retry: int = 10) -> None:
        super().__init__(model, prompt_template, max_retry)
    
    def clean(self,src_path:str,dst_path:str,start=1):
        with open(src_path,"r") as f:
            json_list = list(f)
            unique_cnt,varied_cnt = 0,0
            for json_str in tqdm(json_list[start-1:],desc='Clean'):
                data = json.loads(json_str)
                q = data["question"]
                for idx in range(self.max_retry):
                    try:
                        message = self.PROMPT_TEMPLATE['CLEAN'].format(question = q)
                        print('\033[92m'+"Message below:")
                        print(message)
                        compeletion,answer = self.CreateWrapper(message=message)
                        print('\033[93m'+"Answer below:")
                        print(answer)
                        print('\033[0m')
                        regex = r"\[(.*?)\]"
                        match = re.findall(regex,answer)[-1]
                        print(match)
                        data["valid"] = match
                        if match == ("varied" or "Varied"):
                            varied_cnt += 1
                        elif match == ("unique" or "Unique"):
                            unique_cnt += 1
                        break
                    except Exception as e:
                        print('Error:',e)
                        time.sleep(1)
                        continue
                with open(dst_path,"a") as p:
                    json.dump(data,p)
                    p.write('\n')

            print(f'unique={unique_cnt}')
            print(f'varied={varied_cnt}')