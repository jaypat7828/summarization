import requests

def test_summarization():
    url = "http://127.0.0.1:7860/api/predict"  # Adjust the URL if necessary
    input_text = "This is a test input to check the summarization functionality."
    
    response = requests.post(url, json={
        "data": [input_text]
    })
    
    assert response.status_code == 200
    summary = response.json()["data"][0]
    print("Summary:", summary)

if __name__ == "__main__":
    test_summarization()
