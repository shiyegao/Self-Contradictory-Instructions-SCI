import openai
import copy
import yaml
import time
import json
from tqdm import tqdm

PROMPT_TEMPLATE = yaml.load(open('prompt.yaml'),Loader=yaml.FullLoader)


def list_to_string(my_list):
    # Define the delimiter
    delimiter = ", "
    # Join the list elements with the delimiter and a conditional expression
    my_string = delimiter.join(element if i < len(my_list) - 1 else "and " + element for i, element in enumerate(my_list))
    # Return the resulting string
    return my_string

class LLMObject():
    def __init__(self,model,prompt_template,max_retry:int=10,sys_prompt:str=None) -> None:
        self.model = model
        self.system_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.client = openai.OpenAI(
                                    base_url=openai.base_url,
                                    api_key=openai.api_key
                                )
        
        if sys_prompt:
            self.system_messages[0]['content'] = sys_prompt
        
        self.user_prompt = {"role":"user","content":""}
        self.max_retry = max_retry
        self.PROMPT_TEMPLATE = prompt_template
    
    def CreateChat(self,message:str,temperature:float=1):
        """
        Use this when using gpt-3.5-turbo.
          text = completion['choices'][0]['message']['content']
        """
        messages = copy.deepcopy(self.system_messages)
        
        self.user_prompt["content"]= message
        messages.append(self.user_prompt)
        
        
        for _ in range(self.max_retry):
            try:
                completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        )
                print(completion)
                break
            except Exception as e:
                print(f"[OPENAI WARNING] An exception occurred: {e}, sleep for 5s")
                time.sleep(5)
                continue
        return completion
    
    def CreateCompletion(self,message:str,temperature:float=0,max_tokens=600):
        """
          Use this when using gpt-3.5-turbo-instruct.
          text = completion['choices'][0]['text']
        """
        for _ in range(self.max_retry):
            try:
                completion = openai.Completion.create(
                    model=self.model,
                    prompt=message,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    )
                break
            except Exception as e:
                print(f"[OPENAI WARNING] An exception occurred, sleep for 5s")
                time.sleep(5)
                continue
        return completion
    
    def ZhipuCreateChat(self,message:str,temperature:float=0):
        """
          Use this when using zhipu.
          text = completion['choices'][0]['message']['content']
        """
        messages = copy.deepcopy(self.system_messages)
        self.user_prompt["content"]= message
        messages.append(self.user_prompt)
        for _ in range(self.max_retry):
            completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    )
            print(completion)
            try:
                print(f"[ZHIPU WARNING] An exception {completion.error['type']}occurred, sleep for 5s")
                print(completion)
                time.sleep(5)
                continue
            except:
                break
        return completion
    
    def MutliRoundChat(self,messages:list,temperature:float=1):
        """
        Specially designed for attributes extraction, which needs conversation histroy
          text = completion['choices'][0]['message']['content']
        """        
        for _ in range(self.max_retry):
            completion = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    )
                
            try:
                print(f"[ZHIPU WARNING] An exception {completion['error']['type']}occurred, sleep for 5s")
                time.sleep(5)
                continue
            except:
                break
        return completion,completion['choices'][0]['message']['content']
    
    
    def CreateWrapper(self,message:str,temperature:float=1):
        if self.model == "gpt-3.5-turbo":
            completion = self.CreateChat(message=message,temperature=temperature)
            return completion, completion['choices'][0]['message']['content']
        elif self.model == "gpt-3.5-turbo-instruct":
            completion = self.CreateCompletion(message=message)
            return completion, completion['choices'][0]['text']
        elif self.model == "chatglm_turbo":
            completion = self.ZhipuCreateChat(message=message,temperature=temperature)
            return completion, completion.choices[0].message.content


class conflict_dataset:
    def __init__(self,dataset_type,dataset_path):
        self.type = dataset_type
        self.dataset_path = dataset_path
        self.length = 0
        assert(self.type in [1,2,3,4])
        type_description=['1.rule violation conflict',
                          '2.object attribute conflict',
                          '3.exclusive instrution conflict',
                          '4.forbidden word conflict']
        print('current dataset:',type_description[self.type-1])
        self.parse_jsonl()
        
    def parse_jsonl(self):
        self.json_list = list(open(self.dataset_path,'r'))
        self.length = len(self.json_list)
        print(f'{self.length} lines in total')
        
    def get_dict(self,index:int=0):
        """get dataset[index]"""
        data = json.loads(self.json_list[index])
        return data
        
    def get_prompt(self,index:int=0):
        data = self.get_dict(index)
        if self.type == 1:
            prompt =  data['context']+data['violation']+data['question']
            return prompt
        elif self.type == 2:
            prompt_list = []
            for conflict in data['dataset']:
                prompt =data['original text'] + conflict['opposite description'] +'\n'+conflict['question']
                prompt_list.append(prompt)
                
            return prompt_list
                
        elif self.type == 3:
            return data['content']
        elif self.type == 4:
            return data['prompt']
        
        
class LLM_answer(LLMObject):
    def __init__(self, model, prompt_template, max_retry: int = 10, sys_prompt: str = None) -> None:
        super().__init__(model, prompt_template, max_retry, sys_prompt)
        
    def answer(self,dataset:conflict_dataset,result_path:str,start:int=1):
        self.dataset = dataset
        for index in tqdm(range(self.dataset.length)[start-1:],desc='answer'):
            data = self.dataset.get_dict(index)
            prompt = self.dataset.get_prompt(index)
            _,answer = self.CreateWrapper(message=prompt)
            data['answer']=answer
            with open(result_path,'a') as f:
                json.dump(data,f)
                f.write('\n')
                
                
class LLM_evaluation(LLMObject):
    def __init__(self, model, prompt_template, max_retry: int = 10, sys_prompt: str = None) -> None:
        super().__init__(model, prompt_template, max_retry, sys_prompt)
        self.evaluation_prompt ='''We are trying to test LLM's ability to detect conflict in user prompt. The LLM is given a prompt that is self-contradictory. The user's prompt may contain two mutually exclusive instructions for LLM, meaning that both instructions cannot be followed simultaneously. Sometimes, the user's prompt may provide contradictory information and give LLM an unanswerable question due to the information conflict.
Here is an LLM reply to a self-contradictory user prompt. I need you to judge if LLM have detected the conflict in user prompt. Your reply should  be strictly in the format like "[Yes] \n the reason why LLM had detected the conflict in user prompt". For example:
"[Yes]\nLLM has mentioned that there are two sentences that talk about the color of the dream horse. But they are not consistent." 
"[No]\n LLM simply do what the user asked and didn't notice the conflict"
"[Yes]\n LLM apologized for not being able to do two tasks simultaneously"

## LLM Answer
{LLM_reply}
'''
        
    def evaluate(self,dataset:conflict_dataset,result_path:str,start:int=1):
        self.dataset = dataset
        for index in tqdm(range(self.dataset.length)[start-1:],desc='evaluate'):
            data = self.dataset.get_dict(index)
            LLM_reply = data['answer']
            prompt = self.evaluation_prompt.format(LLM_reply=LLM_reply)
            _,evaluation = self.CreateWrapper(message=prompt)
            data['evaluation']=evaluation
            with open(result_path,'a') as f:
                json.dump(data,f)
                f.write('\n')