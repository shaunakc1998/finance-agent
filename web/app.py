# web/app.py

import os
import uuid
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
import sqlite3
import sys

# Add parent directory to path to import finance agent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_agent import (
    llm, tools, agent, agent_executor,
    RunnableWithMessageHistory, ChatMessageHistory
)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Database setup
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database', 'finance_chat.db')

def init_db():
    """Initialize the database with required tables"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        user_id TEXT,
        name TEXT,
        created_at TIMESTAMP
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        timestamp TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
    )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

# Initialize database on startup
init_db()

# Routes
@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all chat sessions"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM sessions ORDER BY created_at DESC')
    sessions = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify({'sessions': sessions})

@app.route('/api/sessions', methods=['POST'])
def create_session():
    """Create a new chat session"""
    data = request.json
    session_id = str(uuid.uuid4())
    user_id = data.get('user_id', 'anonymous')
    name = data.get('name', f'Chat {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO sessions VALUES (?, ?, ?, ?)', 
              (session_id, user_id, name, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    return jsonify({'session_id': session_id, 'name': name})

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session and its messages"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
    c.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/sessions/<session_id>/rename', methods=['POST'])
def rename_session(session_id):
    """Rename a chat session"""
    data = request.json
    new_name = data.get('name', f'Chat {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE sessions SET name = ? WHERE session_id = ?', (new_name, session_id))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'name': new_name})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a chat message with the finance agent"""
    data = request.json
    session_id = data.get('session_id')
    message = data.get('message')
    
    if not session_id or not message:
        return jsonify({'error': 'Missing session_id or message'}), 400
    
    # Retrieve chat history for this session
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp', 
              (session_id,))
    history = [dict(row) for row in c.fetchall()]
    
    # Save the user message
    now = datetime.now().isoformat()
    c.execute('INSERT INTO messages VALUES (NULL, ?, ?, ?, ?)', 
              (session_id, 'user', message, now))
    conn.commit()
    
    # Process with the finance agent
    response = process_with_finance_agent(message, history)
    
    # Save the assistant response
    c.execute('INSERT INTO messages VALUES (NULL, ?, ?, ?, ?)', 
              (session_id, 'assistant', response, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    return jsonify({'response': response})

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Get chat history for a session"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp', 
              (session_id,))
    messages = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return jsonify({'messages': messages})

def process_with_finance_agent(user_message, history):
    """Process a message with the finance agent"""
    # Convert history to the format expected by LangChain
    chat_history = ChatMessageHistory()
    
    for msg in history:
        if msg['role'] == 'user':
            chat_history.add_user_message(msg['content'])
        elif msg['role'] == 'assistant':
            chat_history.add_ai_message(msg['content'])
    
    # Create agent with history
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    # Process the message
    try:
        # Disable terminal formatting for web responses
        response = agent_with_chat_history.invoke(
            {"input": user_message},
            {"configurable": {"session_id": "default"}}
        )
        
        # Clean up any ANSI color codes that might be in the response
        output = response['output']
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned_output = ansi_escape.sub('', output)
        
        return cleaned_output
    except Exception as e:
        print(f"Error processing message: {e}")
        return f"I encountered an error while processing your request: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5001)
