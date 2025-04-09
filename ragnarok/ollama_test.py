from ollama import Client
client = Client(
  host='http://10.0.0.11:11434',
  headers={}
)
response = client.chat(model='llama3.2', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])