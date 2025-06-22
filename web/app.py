# web/app.py

import os
import uuid
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, make_response
import sqlite3
import sys

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Add parent directory to path to import finance agent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_agent import (
    llm, tools, agent, agent_executor,
    RunnableWithMessageHistory, ChatMessageHistory
)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Database setup
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finance_chat.db')

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
    """Render the default guided financial planning interface"""
    return render_template('guided_chat.html')

@app.route('/settings')
def settings():
    """Render the settings page"""
    # Get current API keys for the user
    user_id = request.cookies.get('user_id', 'anonymous')
    current_keys = get_user_api_keys(user_id)
    
    response = make_response(render_template('settings.html', current_keys=current_keys))
    
    # Set user_id cookie if not present
    if not request.cookies.get('user_id'):
        import uuid
        new_user_id = f'user_{uuid.uuid4().hex[:16]}'
        response.set_cookie('user_id', new_user_id, max_age=365*24*60*60)  # 1 year
    
    return response

@app.route('/save_api_keys', methods=['POST'])
def save_api_keys():
    """Save API keys for a user"""
    user_id = request.cookies.get('user_id', 'anonymous')
    
    # Get API keys from form
    api_keys = {
        'openai_api_key': request.form.get('openai_api_key', ''),
        'alpha_vantage_key': request.form.get('alpha_vantage_key', '')
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
    conversation_state = data.get('conversation_state')
    
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
    
    # Process with the finance agent and get both thinking and final response
    thinking_response, final_response = process_with_finance_agent_separate(message, history, conversation_state)
    
    # Save both responses as separate messages
    c.execute('INSERT INTO messages VALUES (NULL, ?, ?, ?, ?)', 
              (session_id, 'assistant_thinking', thinking_response, datetime.now().isoformat()))
    c.execute('INSERT INTO messages VALUES (NULL, ?, ?, ?, ?)', 
              (session_id, 'assistant', final_response, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    return jsonify({
        'thinking_response': thinking_response,
        'final_response': final_response
    })

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

def format_response_with_thinking(response, thinking_steps):
    """Format the response with thinking process"""
    if not thinking_steps:
        return response
    
    # Create thinking process section
    thinking_section = "\n\n<details>\n<summary><strong>🧠 Analysis Process</strong> (Click to expand)</summary>\n\n"
    
    for i, step in enumerate(thinking_steps, 1):
        tool_name = step['tool']
        reasoning = step['reasoning']
        
        # Map tool names to user-friendly names
        tool_display_names = {
            'fundamentals_tool': '📊 Financial Fundamentals',
            'technicals_tool': '📈 Technical Analysis',
            'financial_insights_tool': '📰 Recent Earnings & News',
            'industry_comparison_tool': '🏢 Industry Comparison',
            'economic_factors_tool': '🌍 Economic Factors',
            'visualization_tool': '📊 Data Visualization',
            'forecast_tool': '🔮 Price Forecasting',
            'product_search': '🔍 Product Research'
        }
        
        display_name = tool_display_names.get(tool_name, f"🔧 {tool_name}")
        
        thinking_section += f"**Step {i}: {display_name}**\n"
        thinking_section += f"*{reasoning}*\n\n"
        
        # Add a brief summary of what was found (first 200 chars of output)
        if 'output' in step and step['output']:
            output_preview = step['output'][:200].replace('\n', ' ').strip()
            if len(step['output']) > 200:
                output_preview += "..."
            thinking_section += f"✅ *Retrieved: {output_preview}*\n\n"
        
        thinking_section += "---\n\n"
    
    thinking_section += "</details>\n\n"
    
    # Combine thinking process with response
    return thinking_section + response

def process_with_finance_agent_separate(user_message, history, conversation_state=None):
    """Process a message with the finance agent and return thinking and final response separately"""
    # Get user API keys
    user_id = request.cookies.get('user_id', 'anonymous')
    api_keys = get_user_api_keys(user_id)
    
    # Convert history to the format expected by LangChain
    chat_history = ChatMessageHistory()
    
    for msg in history:
        if msg['role'] == 'user':
            chat_history.add_user_message(msg['content'])
        elif msg['role'] == 'assistant' or msg['role'] == 'assistant_thinking':
            chat_history.add_ai_message(msg['content'])
    
    # Set OpenAI API key if provided by user, otherwise use environment variable
    openai_api_key = api_keys.get('openai_api_key', None)
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Create a new LLM instance with the correct API key
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_openai_functions_agent, AgentExecutor
        
        # Create new LLM with the correct API key
        custom_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=openai_api_key,
            streaming=True,
            max_tokens=1500,
            request_timeout=30
        )
        
        # Create new agent with the custom LLM
        from chat_agent import tools, prompt
        custom_agent = create_openai_functions_agent(custom_llm, tools, prompt)
        custom_agent_executor = AgentExecutor(
            agent=custom_agent, 
            tools=tools, 
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=20,
            max_execution_time=120
        )
        
        # Use the custom agent executor
        agent_executor_to_use = custom_agent_executor
    else:
        # Fall back to the default agent executor
        from chat_agent import agent_executor
        agent_executor_to_use = agent_executor
    
    # Monkey patch the technicals_tool to use user API keys
    from tools import technicals
    original_technicals_tool = technicals.get_technicals

    def patched_technicals(ticker):
        alpha_vantage_key = api_keys.get('alpha_vantage_key', None)
        return original_technicals_tool(ticker, user_api_key=alpha_vantage_key)
    
    # Temporarily replace the function
    technicals.get_technicals = patched_technicals
    
    # Create agent with history
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor_to_use,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    # Create thinking response based on the input
    input_text = user_message.lower()
    thinking_response = "🧠 **THINKING PROCESS**\n\n"
    
    if any(word in input_text for word in ['company', 'stock', 'ticker', 'analyze', 'research', 'earning', 'call']):
        thinking_response += """**Step 1: Financial Fundamentals Analysis**
*I'll retrieve comprehensive financial data including P/E ratio, revenue, debt, profitability metrics, and key ratios for detailed fundamental analysis.*

**Step 2: Recent Earnings & News Analysis**
*I'll gather recent earnings call transcripts, management commentary, and strategic updates to understand current business performance and market sentiment.*

**Step 3: Industry Comparison**
*I'll compare valuation metrics, performance, and competitive positioning against industry peers to assess relative value and market position.*

**Step 4: Technical Analysis**
*I'll analyze price trends, support/resistance levels, and technical indicators to understand market momentum and entry/exit points.*

**Step 5: Investment Synthesis**
*I'll combine all analysis to provide a comprehensive investment recommendation with specific risk assessment and position sizing guidance.*"""
    
    elif any(word in input_text for word in ['etf', 'portfolio', 'diversif', 'invest']):
        thinking_response += """**Step 1: ETF Fundamentals Analysis**
*I'll analyze ETF fundamentals, expense ratios, and performance metrics for portfolio construction.*

**Step 2: Economic Factors Assessment**
*I'll analyze macroeconomic factors that could impact portfolio performance and asset allocation decisions.*

**Step 3: Diversification Strategy**
*I'll evaluate different asset classes and geographic exposure to optimize risk-adjusted returns.*

**Step 4: Cost Analysis**
*I'll compare expense ratios and tax efficiency across different ETF options to maximize net returns.*"""
    
    elif any(word in input_text for word in ['save', 'goal', 'plan', 'buy']):
        thinking_response += """**Step 1: Product Research**
*I'll research product pricing and associated costs to determine total savings target.*

**Step 2: Financial Capacity Analysis**
*I'll analyze available income and expenses to determine realistic savings capacity.*

**Step 3: Investment Strategy**
*I'll analyze investment options and expected returns for the savings goal timeline.*

**Step 4: Implementation Plan**
*I'll create a detailed month-by-month savings and investment plan with specific actionable steps.*"""
    
    else:
        thinking_response += """**Step 1: Query Analysis**
*I'll analyze your question to determine the most relevant financial tools and data sources needed.*

**Step 2: Data Gathering**
*I'll retrieve the necessary financial data, market information, and analysis tools.*

**Step 3: Analysis & Synthesis**
*I'll process the information and provide comprehensive insights tailored to your specific question.*"""
    
    thinking_response += "\n\n*Processing your request...*"
    
    # Process the message for final response
    try:
        # Process the message
        response = agent_with_chat_history.invoke(
            {"input": user_message},
            {"configurable": {"session_id": "default"}}
        )
        
        # Clean up any ANSI color codes that might be in the response
        output = response['output']
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned_output = ansi_escape.sub('', output)
        
        # Format final response
        final_response = f"📊 **FINAL ANALYSIS**\n\n{cleaned_output}"
        
        # Restore original function
        technicals.get_technicals = original_technicals_tool
        
        return thinking_response, final_response
        
    except Exception as e:
        print(f"Error processing message: {e}")
        error_response = f"I encountered an error while processing your request: {str(e)}"
        return thinking_response, error_response

def process_with_finance_agent(user_message, history, conversation_state=None):
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
    
    # Set OpenAI API key if provided by user, otherwise use environment variable
    openai_api_key = api_keys.get('openai_api_key', None)
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Create a new LLM instance with the correct API key
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_openai_functions_agent, AgentExecutor
        
        # Create new LLM with the correct API key
        custom_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=openai_api_key,
            streaming=True,
            max_tokens=1500,
            request_timeout=30
        )
        
        # Create new agent with the custom LLM
        from chat_agent import tools, prompt
        custom_agent = create_openai_functions_agent(custom_llm, tools, prompt)
        custom_agent_executor = AgentExecutor(
            agent=custom_agent, 
            tools=tools, 
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=20,
            max_execution_time=120
        )
        
        # Use the custom agent executor
        agent_executor_to_use = custom_agent_executor
    else:
        # Fall back to the default agent executor
        from chat_agent import agent_executor
        agent_executor_to_use = agent_executor
    
    # Monkey patch the technicals_tool to use user API keys
    from tools import technicals
    original_technicals_tool = technicals.get_technicals

    def patched_technicals(ticker):
        alpha_vantage_key = api_keys.get('alpha_vantage_key', None)
        return original_technicals_tool(ticker, user_api_key=alpha_vantage_key)
    
    # Temporarily replace the function
    technicals.get_technicals = patched_technicals
    
    # Create agent with history
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor_to_use,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    # Create a simple thinking tracker based on the input
    thinking_steps = []
    
    # Analyze the input to determine what tools will likely be used
    input_text = user_message.lower()
    
    if any(word in input_text for word in ['company', 'stock', 'ticker', 'analyze', 'research']):
        thinking_steps.extend([
            {
                'tool': 'fundamentals_tool',
                'reasoning': 'Retrieving comprehensive financial data including P/E ratio, revenue, debt, profitability metrics, and key ratios for detailed fundamental analysis.',
                'output': 'Financial fundamentals data retrieved successfully'
            },
            {
                'tool': 'financial_insights_tool', 
                'reasoning': 'Gathering recent earnings call transcripts, management commentary, and strategic updates to understand current business performance.',
                'output': 'Recent earnings and news data retrieved'
            },
            {
                'tool': 'industry_comparison_tool',
                'reasoning': 'Comparing valuation metrics, performance, and competitive positioning against industry peers to assess relative value.',
                'output': 'Industry comparison data retrieved'
            }
        ])
    elif any(word in input_text for word in ['etf', 'portfolio', 'diversif', 'invest']):
        thinking_steps.extend([
            {
                'tool': 'fundamentals_tool',
                'reasoning': 'Analyzing ETF fundamentals, expense ratios, and performance metrics for portfolio construction.',
                'output': 'ETF fundamental data retrieved'
            },
            {
                'tool': 'economic_factors_tool',
                'reasoning': 'Analyzing macroeconomic factors that could impact portfolio performance and asset allocation decisions.',
                'output': 'Economic factors analysis completed'
            }
        ])
    elif any(word in input_text for word in ['save', 'goal', 'plan', 'buy']):
        thinking_steps.extend([
            {
                'tool': 'product_search',
                'reasoning': 'Researching product pricing and associated costs to determine total savings target.',
                'output': 'Product pricing research completed'
            },
            {
                'tool': 'fundamentals_tool',
                'reasoning': 'Analyzing investment options and expected returns for savings goal timeline.',
                'output': 'Investment analysis for savings goal completed'
            }
        ])
    
    # Process the message
    try:
        # If we have conversation state, add it to the message for context
        input_message = user_message
        
        if conversation_state and conversation_state.get('path'):
            # Add conversation state to the message for better context
            path = conversation_state.get('path')
            mode = conversation_state.get('mode', 'flexible')
            needs_personalization = conversation_state.get('needsPersonalization', False)
            
            # Create context-aware prompts based on the conversation path
            if path == 'research_company':
                if needs_personalization:
                    input_message = f"""The user is asking about company research and wants personalized investment advice. 

User message: {user_message}

INSTRUCTIONS:
1. If the user has provided financial information (income, savings, investment amount, etc.), use it to give personalized advice
2. If they haven't provided enough financial context, provide general company analysis but suggest they could get personalized recommendations
3. Always include:
   - Company fundamental analysis (financials, valuation, competitive position)
   - Technical analysis if relevant
   - General investment considerations
   - Risk assessment
4. If giving personalized advice, include specific position sizing and risk management
5. Be conversational and natural - don't use rigid formatting unless specifically helpful

Focus on being helpful and comprehensive while maintaining a natural conversation flow."""
                else:
                    input_message = f"""The user is asking about company research. Provide comprehensive analysis but keep it general unless they specifically ask for personalized advice.

User message: {user_message}

INSTRUCTIONS:
1. Provide thorough company analysis including:
   - Business overview and competitive position
   - Financial health and key metrics
   - Recent performance and trends
   - Investment thesis (bull/bear cases)
   - Risk factors
2. Keep recommendations general unless user provides personal financial context
3. If appropriate, mention that personalized recommendations would require knowing their financial situation
4. Be conversational and engaging
5. Use specific data and examples when possible"""
                    
            elif path == 'explore_etfs':
                if needs_personalization:
                    input_message = f"""The user is asking about ETFs and wants personalized portfolio recommendations.

User message: {user_message}

INSTRUCTIONS:
1. If the user has provided financial information, create a specific portfolio recommendation
2. If they haven't provided enough context, ask for key information naturally in conversation
3. Always include:
   - Specific ETF recommendations with ticker symbols and expense ratios
   - Asset allocation suggestions
   - Implementation strategy (brokerages, automation)
   - Rebalancing approach
4. If giving personalized advice, include specific dollar amounts and timelines
5. Consider tax efficiency and account types
6. Be practical and actionable"""
                else:
                    input_message = f"""The user is asking about ETFs. Provide helpful general guidance.

User message: {user_message}

INSTRUCTIONS:
1. Provide comprehensive ETF guidance including:
   - Popular ETF categories and examples
   - How to build diversified portfolios
   - Cost considerations (expense ratios)
   - Tax efficiency
   - Implementation tips
2. Give specific ETF recommendations with ticker symbols
3. Explain concepts clearly for different experience levels
4. If appropriate, mention that personalized portfolios would require knowing their situation
5. Be educational and practical"""
            elif path == 'savings_goal':
                # Enhanced savings goal with comprehensive financial planning
                goal_item = inputs.get('goal_item', 'Unknown item')
                monthly_income = inputs.get('monthly_income', 0)
                monthly_expenses = inputs.get('monthly_expenses', 0)
                
                # Use product search tool to get pricing information
                from tools.product_search import search_product_price, format_price_info
                
                # Determine product type based on goal item
                product_type = "general"
                if any(word in goal_item.lower() for word in ["car", "vehicle", "auto", "hyundai", "toyota", "honda", "ford", "bmw", "mercedes", "audi", "tesla"]):
                    product_type = "car"
                elif any(word in goal_item.lower() for word in ["house", "home", "property", "real estate"]):
                    product_type = "house"
                elif any(word in goal_item.lower() for word in ["phone", "iphone", "samsung", "laptop", "computer", "electronics"]):
                    product_type = "electronics"
                
                # Get product pricing information
                price_info = search_product_price(goal_item, product_type)
                formatted_price_info = format_price_info(price_info)
                
                input_message = f"""I need a comprehensive financial plan for saving toward a specific goal. Here are the details:

{user_message}

PRODUCT PRICING RESEARCH:
{formatted_price_info}

IMPORTANT REQUIREMENTS:
1. FINANCIAL ANALYSIS: Based on the monthly income of ${monthly_income} and monthly expenses of ${monthly_expenses}, calculate:
   - Available monthly savings capacity
   - Recommended emergency fund requirements
   - Debt payoff priorities if applicable
   - Net disposable income for savings/investments

2. GOAL-SPECIFIC PLANNING: Using the product pricing information above, create a detailed plan that includes:
   - Total amount needed (including all additional costs)
   - Timeline to reach the goal based on available savings capacity
   - Whether to save in cash vs invest based on timeline
   - Monthly savings target to meet the goal

3. INVESTMENT STRATEGY: Provide specific investment recommendations including:
   - Exact ETF/stock ticker symbols with expense ratios
   - Asset allocation percentages based on timeline and risk tolerance
   - Expected annual returns based on historical data
   - Risk assessment for each recommendation
   - Specific brokerage recommendations (Fidelity, Vanguard, Schwab, etc.)

4. DETAILED SAVINGS PLAN: Create a month-by-month plan showing:
   - How much to save in high-yield savings vs invest
   - Specific investment amounts for each recommended vehicle
   - Timeline milestones and progress tracking
   - Contingency plans if markets underperform or income changes

5. ACTIONABLE STEPS: Provide specific next steps the user should take, including:
   - Which accounts to open (high-yield savings, brokerage, etc.)
   - How to set up automatic transfers and investments
   - When to rebalance the portfolio
   - Apps or tools for tracking progress

6. OPTIMIZATION STRATEGIES: Include advanced strategies such as:
   - Tax-advantaged accounts if applicable (401k, IRA, HSA)
   - Dollar-cost averaging vs lump sum investing
   - Tax-loss harvesting opportunities
   - Ways to increase income or reduce expenses

Please make this as comprehensive and actionable as possible, with real numbers, specific recommendations, and a clear step-by-step implementation plan."""
        
        # Disable terminal formatting for web responses
        response = agent_with_chat_history.invoke(
            {"input": input_message},
            {"configurable": {"session_id": "default"}}
        )
        
        # Clean up any ANSI color codes that might be in the response
        output = response['output']
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned_output = ansi_escape.sub('', output)
        
        # Format the response with thinking process
        formatted_response = format_response_with_thinking(cleaned_output, thinking_steps)
        
        # Restore original function
        technicals.get_technicals = original_technicals_tool
        
        return formatted_response
    except Exception as e:
        print(f"Error processing message: {e}")
        return f"I encountered an error while processing your request: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
