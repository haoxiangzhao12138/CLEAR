from openai import OpenAI
import os
import time

start_time = time.time()

os.environ["OPENAI_API_KEY"] = "sk-zSaDTLLv9cSEwRRt9oLhmMMNpFidKy4cGEtogaICub4mFw67"
os.environ["OPENAI_API_BASE"] = "http://yy.dbh.baidu-int.com/v1"

def call_model_openai(messages, modelname):
    k = 1
    ouput = ""
    while(k > 0):
        k -= 1
        try:
            client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["OPENAI_API_BASE"],
            )
            completion = client.chat.completions.create(
                model=modelname,
                messages=messages,
                top_p=0.8,
                temperature=0.8,
                max_tokens=16384,
                #stream=True,
                #extra_body={"enable_thinking":"true"},
            )
        
            print(completion)
            ouput = completion.choices[0].message.content
             
            #ouput = completion.choices[0].message


            if ouput != None and ouput != "":
                break
        except Exception as e:
            print(e)
            continue


    return ouput, None

query = "say hello"
messages = []
messages.append({"role": "user", "content": query})
output,_ = call_model_openai(messages, "gpt-4.1")

end_time = time.time()

print(end_time - start_time)

print(output)