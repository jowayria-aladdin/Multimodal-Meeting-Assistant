import json
import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen2.5:7b",
    "prompt": """Return JSON only.
{
    "message": "hello"
}""",
    "stream": False,
    "format": "json",
    "options": {
        "temperature": 0.1
    }
}

response = requests.post(url, json=payload, timeout=120)

print("Status:", response.status_code)
print("\nRaw response:")
print(response.text)

if response.status_code == 200:
    data = response.json()
    print("\nParsed Ollama response:")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    if "response" in data:
        print("\nModel text:")
        print(data["response"])

        try:
            parsed_model_json = json.loads(data["response"])
            print("\nParsed model JSON:")
            print(json.dumps(parsed_model_json, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("\nModel did not return valid JSON.")
else:
    print("\nRequest failed.")