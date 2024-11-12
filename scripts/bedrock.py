import boto3
import json

f = open('request.txt', 'r')
data = f.read()

# Create the bedrock client with specified region
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 2000, # Maximum number of tokens (words/subwords) in the generated response (0 - 4096)
    "temperature": 1.0,  # Controls the randomness/creativity of the generated text (0.0 - 1.0 higher values are more random)
    "top_p": 0.999,  # Controls the nucleus sampling, which filters out unlikely word choices (0.0 - 1.0, higher values are more inclusive)
    "top_k": 250,  # Controls the top-k sampling, which filters out unlikely word choices (0 - 500, 0 means no filtering)
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": f"{data}"
          }
        ]
      }
    ]
})

modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
accept = 'application/json'
contentType = 'application/json'

response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

response_body = json.loads(response.get('body').read())

content = response_body.get('content', [])
if content:
    text = content[0].get('text', '')
    print(text)
else:
    print("No content found in the response.")
