import os

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# Find API Key
load_dotenv(find_dotenv())

# Make sure you have a .env file with your API key. It can also be passed using the 'api_key' parameter, but be careful to not check it in.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)
