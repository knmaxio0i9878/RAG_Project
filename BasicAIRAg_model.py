from openai import OpenAI
import pandas as ps
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your .env file.")
client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key=api_key
)
MODEL = "nvidia/nemotron-3.5-lightning:free"
print("Setup Completed")
role = input("Enter the Role:")
content = input("Enter the Prompt:")

response = client.chat.completions.create(
    model = MODEL,
    messages = [{"role":role,"content":content}]
)
print(response.choices[0].message.content)

response = client.chat.completions.create(
    model = MODEL,
    messages = [{"role":"system","content":"You are an expert python teacher.Explain Everything in simple examples"},
     {"role":"user","content":"Explain List Comprehension"}
                ]
)
print(response.choices[0].message.content)
while True:
  question = input("You:")
  if question.lower() == "exit":
    break
  response = client.chat.completions.create(
      model=MODEL,
      messages=[{"role":"user","content":question}]
  )
  print("AI:",response.choices[0].message.content)

  while True:
    question = input("You:")
    if question.lower() == "exit":
        break
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":question}]
    )
    answer = response.choices[0].message.content
    print("AI:",answer)