from openai import OpenAI

# client = OpenAI(
#   api_key="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"
# )
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "write a haiku about ai"}
    ],
    temperature= 0.0,
    max_tokens= 2000,
    top_p= 0.95,
    frequency_penalty= 0,
    presence_penalty= 0,
)

print(completion.choices[0].message)