#!/usr/bin/env python3
# test_api.py - Test script for the finance agent API

import requests
import json
import sys

def create_session():
    """Create a new chat session"""
    url = "http://localhost:5001/api/sessions"
    payload = {
        "user_id": "test_user",
        "name": "API Test Session"
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error creating session: {e}")
        sys.exit(1)

def send_message(session_id, message):
    """Send a message to the finance agent"""
    url = "http://localhost:5001/api/chat"
    payload = {
        "session_id": session_id,
        "message": message
    }
    
    try:
        print(f"Sending message: {message}")
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending message: {e}")
        sys.exit(1)

def get_history(session_id):
    """Get chat history for a session"""
    url = f"http://localhost:5001/api/history/{session_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting history: {e}")
        sys.exit(1)

def main():
    # Create a new session
    print("Creating new session...")
    session = create_session()
    session_id = session["session_id"]
    print(f"Session created with ID: {session_id}")
    
    # Test messages
    test_messages = [
        "What is the current stock price of AAPL?",
        "What is the current stock price of MCHP?",
        "What is the current stock price of AI?",
        "What is the current stock price of NONEXISTENT?"
    ]
    
    # Send each test message and print the response
    for message in test_messages:
        print("\n" + "="*50)
        response = send_message(session_id, message)
        print("\nResponse:")
        print(response["response"])
        print("="*50)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
