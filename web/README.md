# Finance Agent Chat UI

A ChatGPT-like web interface for the finance agent with persistent chat history.

## Features

- Modern web interface similar to ChatGPT
- Persistent chat history stored in SQLite database
- Multiple chat sessions with ability to rename and delete
- Markdown rendering for rich text formatting
- Code syntax highlighting
- Mobile-responsive design
- Example prompts for quick start

## Prerequisites

- Python 3.8+
- Flask
- SQLite3
- All dependencies from the main finance agent

## Installation

1. Make sure you have all the required dependencies installed:

```bash
pip install flask
```

2. The web interface uses the existing finance agent, so ensure that all dependencies for the main project are installed as well.

## Usage

1. Navigate to the web directory:

```bash
cd web
```

2. Run the Flask application:

```bash
python app.py
```

3. Open your browser and go to:

```
http://localhost:5000
```

## How It Works

The web interface consists of:

1. **Backend (app.py)**: A Flask application that:
   - Manages chat sessions and history in a SQLite database
   - Integrates with the existing finance agent
   - Provides RESTful API endpoints for the frontend

2. **Frontend**:
   - HTML/CSS/JavaScript interface
   - Communicates with the backend via API calls
   - Renders chat messages with markdown support
   - Provides a sidebar for managing chat sessions

3. **Database**:
   - SQLite database for storing chat sessions and messages
   - Located at `web/database/finance_chat.db`

## API Endpoints

- `GET /api/sessions`: Get all chat sessions
- `POST /api/sessions`: Create a new chat session
- `DELETE /api/sessions/<session_id>`: Delete a chat session
- `POST /api/sessions/<session_id>/rename`: Rename a chat session
- `POST /api/chat`: Process a chat message
- `GET /api/history/<session_id>`: Get chat history for a session

## Customization

You can customize the appearance by modifying:

- `static/css/styles.css`: Change colors, fonts, and layout
- `templates/index.html`: Modify the HTML structure
- `static/js/chat.js`: Adjust the JavaScript behavior

## Troubleshooting

- If you encounter database errors, try deleting the `web/database/finance_chat.db` file and restarting the application.
- If the chat agent fails to respond, check that the OpenAI API key is properly set in the `.env` file.
- For other issues, check the Flask server logs for error messages.

## Future Improvements

- User authentication
- Export chat history to PDF/CSV
- Dark mode toggle
- Streaming responses
- Voice input/output
- Integration with additional financial data sources
