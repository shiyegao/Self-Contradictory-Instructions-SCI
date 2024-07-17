import openai
import json
import random,base64
import copy
import re,yaml
import time
import itertools as it

PROMPT_TEMPLATE = yaml.load(open('prompt.yaml'),Loader=yaml.FullLoader)


class LLMObject():
    def __init__(self,model,prompt_template,max_retry:int=10) -> None:
        self.model = model
        self.system_messages = [{"role": "system", "content": "You are a helpful assistant."}]
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
        #print(messages)
        for _ in range(self.max_retry):
            try:
                completion = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        )
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
          text = completion['choices'][0]['text']
        """
        messages = copy.deepcopy(self.system_messages)
        self.user_prompt["content"]= message
        messages.append(self.user_prompt)
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

        return completion
        
    def CreateWrapper(self,message:str,temperature:float=1):
        if self.model == "gpt-3.5-turbo" or "gpt-4":
            completion = self.CreateChat(message=message,temperature=temperature)
            return completion, completion['choices'][0]['message']['content']
        elif self.model == "gpt-3.5-turbo-instruct":
            completion = self.CreateCompletion(message=message)
            return completion, completion['choices'][0]['text']
        elif self.model == "chatglm_turbo":
            completion = self.ZhipuCreateChat(message=message,temperature=temperature)
            print(completion)
            return completion, completion['choices'][0]['message']['content']


def list_to_string(my_list):
    # Define the delimiter
    delimiter = ", "
    # Join the list elements with the delimiter and a conditional expression
    my_string = delimiter.join(element if i < len(my_list) - 1 else "and " + element for i, element in enumerate(my_list))
    # Return the resulting string
    return my_string


def generate_combinations(A_range, B_range):
    combinations = []
    for A in A_range:
        for B in B_range:
            combinations.append((A, B))
    return combinations






