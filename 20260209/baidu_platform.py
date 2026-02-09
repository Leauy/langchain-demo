from openai import OpenAI




from openai import OpenAI
client = OpenAI(
    base_url='https://qianfan.baidubce.com/v2',
    api_key='bce-v3/ALTAK-UGrfY9ZZq1ME1LlO2uoDS/19de6e32c9019d22ebd3ab839caae754cb2cb779',
default_headers={'appid': 'app-s4NhzTXi'},
)
response = client.chat.completions.create(
    model="ernie-4.5-turbo-128k",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "推荐一本关于人工智能的书。"}
    ],
    temperature=0.8,
    top_p=0.8,
    extra_body={
        "penalty_score":1,
        "stop":[],
        "web_search":{
            "enable": False,
            "enable_trace": False
        }
    }
)
print(response)