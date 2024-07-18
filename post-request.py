import requests
import json

def evaluate_paper(image_path, csv_path, api_url='http://localhost:5000/evaluate'):
    """
    Send a POST request to the evaluation API with an image and CSV file.
    
    Args:
    image_path (str): Path to the image file
    csv_path (str): Path to the CSV file containing correct answers
    api_url (str): URL of the evaluation API endpoint
    
    Returns:
    dict: JSON response from the API
    """
    # Prepare the files for the POST request
    files = {
        'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg'),
        'csv': ('answers.csv', open(csv_path, 'rb'), 'text/csv')
    }

    try:
        # Send POST request to the API
        response = requests.post(api_url, files=files)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse and return the JSON response
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        # Ensure files are closed
        for file in files.values():
            file[1].close()

# Example usage
if __name__ == "__main__":
    image_path = "sample-api-checking/20240328_145148.jpg"
    csv_path = "sample-api-checking/Marks.csv"
    
    result = evaluate_paper(image_path, csv_path)
    
    if result:
        print("Evaluation Result:")
        print(json.dumps(result, indent=2))
        
        print(f"\nTotal Marks: {result['total_marks']}")
        print("\nPredictions vs Correct Answers:")
        for pred, correct in zip(result['predictions'], result['correct_answers']):
            print(f"Predicted: {pred}, Correct: {str(correct).lower()}")
    else:
        print("Failed to get evaluation result.")