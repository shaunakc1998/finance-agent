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
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE,
        api_keys TEXT,
        created_at TIMESTAMP
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

@app.route('/settings')
def settings():
    """Render the settings page"""
    # Get current API keys for the user
    user_id = request.cookies.get('user_id', 'anonymous')
    current_keys = get_user_api_keys(user_id)
    
    return render_template('settings.html', current_keys=current_keys)

@app.route('/save_api_keys', methods=['POST'])
def save_api_keys():
    """Save API keys for a user"""
    user_id = request.cookies.get('user_id', 'anonymous')
    
    # Get API keys from form
    api_keys = {
        'alpha_vantage_key': request.form.get('alpha_vantage_key', ''),
        'finnhub_key': request.form.get('finnhub_key', ''),
        'polygon_key': request.form.get('polygon_key', ''),
        'iex_key': request.form.get('iex_key', '')
    }
    
    # Save API keys to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute('SELECT id FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    
    if user:
        # Update existing user
        cursor.execute(
            'UPDATE users SET api_keys = ? WHERE id = ?',
            (json.dumps(api_keys), user_id)
        )
    else:
        # Create new user
        cursor.execute(
            'INSERT INTO users (id, api_keys, created_at) VALUES (?, ?, ?)',
            (user_id, json.dumps(api_keys), datetime.now().isoformat())
        )
    
    conn.commit()
    conn.close()
    
    return render_template('settings.html', current_keys=api_keys, message="API keys saved successfully")

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

def get_user_api_keys(user_id):
    """Get API keys for a user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT api_keys FROM users WHERE id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result and result[0]:
        return json.loads(result[0])
    return {}

def process_with_finance_agent(user_message, history):
    """Process a message with the finance agent"""
    # Get user API keys
    user_id = request.cookies.get('user_id', 'anonymous')
    api_keys = get_user_api_keys(user_id)
    
    # Convert history to the format expected by LangChain
    chat_history = ChatMessageHistory()
    
    for msg in history:
        if msg['role'] == 'user':
            chat_history.add_user_message(msg['content'])
        elif msg['role'] == 'assistant':
            chat_history.add_ai_message(msg['content'])
    
    # Monkey patch the technicals_tool to use user API keys
    original_technicals_tool = tools.technicals.get_technicals
    
    def patched_technicals(ticker):
        alpha_vantage_key = api_keys.get('alpha_vantage_key', None)
        return original_technicals_tool(ticker, user_api_key=alpha_vantage_key)
    
    # Temporarily replace the function
    tools.technicals.get_technicals = patched_technicals
    
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
        
        # Restore original function
        tools.technicals.get_technicals = original_technicals_tool
        
        return cleaned_output
    except Exception as e:
        print(f"Error processing message: {e}")
        return f"I encountered an error while processing your request: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
