import json
import random
import time
import re
import itertools as it
import os
from tqdm import tqdm
from utils import LLMObject,list_to_string


class Rule_violation_conflict(LLMObject):
    """ Generate (context, violation, question) triple
    """
    
    def __init__(self, model,prompt_template, max_retry: int = 10,sys_prompt=None,batch=20):
        
        # only this subclass need sys_prompt
        if sys_prompt==None:
            sys_prompt=prompt_template['SYS_PROMPT_1']
        
        
        super().__init__(model=model,prompt_template=prompt_template, max_retry = max_retry,sys_prompt=sys_prompt)
        
    def generate_triple_in_batch(self,num_examples:int=4,dataset_path:str='rule_violation_conflict.jsonl'):
        """ radomly select N examples from dataset\n
            and ask LLM to generate a batch of conflict triples
        """
        
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f'The file {dataset_path} does not exist')
        
        # Load examples
        with open(dataset_path, "r") as example_pool:
            examples = [json.loads(line) for line in example_pool]

        # Randomly select two examples
        random.seed(int(time.time()))
        selected_examples = random.sample(examples, num_examples)   
        
        # Construct the user prompt with the selected examples
        user_prompt = "\n".join([f"{i + 1}. {{\n## Context:\n{example['context']}\n## Violating sentence:\n{example['violation']}\n## Conflict-wise question:\n{example['question']}\n}}"
                                    for i, example in enumerate(selected_examples)])     
        
        
        completion,batch_conflicts = self.CreateWrapper(message=user_prompt)
        triples = self.parse_batch_conflicts(batch_conflicts)
        with open(dataset_path,'a') as jsonl_file:
            for triple in triples:
                jsonl_file.write(json.dumps(triple)+'\n')
                
                
    
    def parse_batch_conflicts(self,batch_conflicts:str):
        """A block is surrounded by a pair of {}    
        """
        block_list = []
        start = 0
        while True:
            start = batch_conflicts.find('{',start)
            if (start == -1):
                break
            end = batch_conflicts.find('}',start)
            if (end == -1):
                break
            block_list.append(batch_conflicts[start:end+1])
            start = end + 1
        
        
        block_dicts = []
        for block in block_list:
            block_dicts.append(self.parse_block(block))
        
        return block_dicts
        
    def parse_block(self,block):
        """ parse a block, return a dict
        Args:
            block (str): a block of conflict triple, including a context, a violation and a question
        """
        block_dict = {}
        block_dict['context'] = block.split('\n')[2].lstrip()
        block_dict['violation'] = block.split('\n')[4].lstrip()
        block_dict['question'] = block.split('\n')[6].lstrip()
        return block_dict


class Object_attribute_conflict(LLMObject):
    """Generate conflict from a description text of a virtual object
       \n1. Generate a description text for a virtual object
       \n2. Extract a description sentence for each attribute
       \n3. Write an opposite description for each description sentence
       \n4. Insert the opposite description into the original text, thus get a conflict text
    """
    def __init__(self, model,prompt_template, max_retry: int = 10,num_attribute:int = 8) -> None:
        super().__init__(model, prompt_template,max_retry)
        self.attributes = ["shape", "function", "history", "color", "size", "location", "material", "time", "temperature", "smell", "taste", "sound", "touch", "emotion", "action", "state"]
        self.attribute_choices = list(it.combinations(self.attributes,num_attribute))
        self.data = {}



    def generate_conflict(self,N = 10,dataset_path = 'Object_attribute_conflict.jsonl'):
        """Generate N objects in total
            \n Each object have {num_attribute} attributes
        """
        random.seed(int(time.time()))
        for attribute_list in tqdm(random.choices(self.attribute_choices,k=N)\
                                    ,desc='generating vitual objects'):
            
            try:
                chat_history = self.generate_original_text(list(attribute_list))
                self.extract_attribute_and_build_conflict(chat_history) 
                
                with open(dataset_path, "a") as f:
                    json.dump(self.data, f)
                    f.write("\n")
            except Exception as e:
                print(f'some error happend:{type(e)}')
                pass
    
    def generate_original_text(self,attribute_list):
        
        self.data["attributes"] = attribute_list
        print('attributes=',attribute_list)
        user_prompt = self.PROMPT_TEMPLATE['VIRTUAL_OBJECT_GENERATE'].format(attributes=list_to_string(attribute_list))
        completion,text = self.CreateWrapper(message=user_prompt)
        print(text)
        print('--------------------')
        object = text.split('\n',1)[0].replace('#',' ').strip()
        original_text = text.split('\n',1)[1].strip()
        self.data["object"] = object
        self.data["original text"] = original_text
        
        chat_round = []
        chat_round.append({"role":"user","content":user_prompt})
        chat_round.append({"role":"assistant","content":text})
        # return this round of chat for the next extraction round
        return chat_round
        
    def extract_attribute_and_build_conflict(self,chat_history:list):
        query_prompt = f'''List all attributes of {self.data['object']} from the given text. Your answer should consist of several lines, each line formatted as "- attribute:Statement of attribute".\n"\nFor example:\n- Length:Very long\n- Taste: It possesses no discernible taste as it embodies celestial energies\n'''
        chat_history.append({"role":"user","content":query_prompt})
        
        completion,extracted_attributes = self.MutliRoundChat(messages=chat_history)
        print(extracted_attributes)
        attribute_conflict_list = []
        new_list = []
        for line in extracted_attributes.split('\n'):
            if line == "":
                continue
            dict = {}
            dict["attribute"] = re.findall(r'-\s(.*?):', line)[0]
            dict["description"] = line.split(':',1)[1].strip()
            
            print('attribute:',dict['attribute'])
            print('description:\n',dict['description'])
            dict["opposite description"],dict['question'] = self.generate_opposite_description(dict['attribute'],dict["description"])
            print(dict)
            attribute_conflict_list.append(dict)
            new_list.append(dict['attribute'])
        print(new_list)
        self.data['dataset'] = attribute_conflict_list
        
            
    def generate_opposite_description(self,attribute, description):
        opposite_prompt = "Generate only one sentence that means exactly the opposite of the given sentence. Do not be verbose.\n" + description
        completion,opposite_description = self.CreateWrapper(message=opposite_prompt)
        #insert_prompt = (f"Insert the given sentence into the given text and output the inserted text.\n## Given Sentence:\n{opposite_prompt}\n## Text:{self.data['original text']}")
        #completion,revised_content = self.CreateWrapper(message=insert_prompt)
        question = "Briefly describe the " + attribute + " of the " + self.data['object'] + " based on the given text.\n"
 
        return opposite_description,question
    
    def clean(self,src_path:str,dst_path:str,start=1):
        with open(src_path,"r") as f:
            json_list = list(f)[start-1:]
            for json_str in tqdm(json_list,desc='Clean'):
                object = json.loads(json_str)        
                name = object["object"]
                text = object['original text']
                attributes = object['attributes']
                dataset = object['dataset']
                check_prompt = 'Did the following text describe the {attribute} of {object}?\nPlease mark the attribute with [yes] or [no].\n## Text\n{text}'
                new_dataset = []
                check_list = []
                for conflict in dataset:
                    message = check_prompt.format(attribute = conflict['attribute'],object=name,text=text)
                    print('\033[92m'+"Message below:")
                    print(message)
                    completion,answer = self.CreateWrapper(message=message)
                    print('\033[93m'+"Answer below:")
                    print(answer)
                    print('\033[0m')
                    
                    if 'yes' in answer.lower():
                        print('yes')
                        new_dataset.append(conflict)
                        check_list.append('yes')
                    elif 'no' in answer.lower():
                        print('no')
                        check_list.append('no')
                        print('Attribute did not show up in the text')
                
                print(check_list)
                object['dataset'] = new_dataset
                with open(dst_path,'a') as f:
                    json.dump(object,f)
                    f.write('\n')





