"""Test client for the RecSys-Lite API."""


import requests


def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    health_response = requests.get(f"{base_url}/health")
    print(f"Health check: {health_response.status_code}")
    print(health_response.json())
    
    # Test recommendations endpoint
    user_id = "U_01"
    k = 5
    rec_response = requests.get(f"{base_url}/recommend?user_id={user_id}&k={k}")
    print(f"\nRecommendations for {user_id}: {rec_response.status_code}")
    if rec_response.status_code == 200:
        data = rec_response.json()
        print(f"User ID: {data['user_id']}")
        print("Recommendations:")
        for i, rec in enumerate(data['recommendations']):
            print(f"  {i+1}. Item: {rec['item_id']}, Score: {rec['score']:.4f}")
    else:
        print(f"Error: {rec_response.text}")

if __name__ == "__main__":
    test_api()