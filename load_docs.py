import requests

pdf_url = "https://arxiv.org/pdf/2409.05591v2"
response = requests.get(pdf_url)

with open("paper.pdf", "wb") as f:
    f.write(response.content)

print("PDF downloaded.")
