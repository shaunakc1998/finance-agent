// Global variables
let currentSessionId = null;
let isProcessing = false;
let conversationState = {
    path: null,           // 'research_company', 'explore_etfs', 'savings_goal'
    mode: 'flexible',     // 'flexible' allows free-form conversation
    inputs: {},           // User inputs collected so far
    needsPersonalization: false  // Whether we need personal financial info
};

// DOM elements
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-btn');
const messagesContainer = document.getElementById('messages');
const chatContainer = document.getElementById('chat-container');
const welcomeScreen = document.getElementById('welcome-screen');
const inputOptions = document.getElementById('input-options');

// Initialize the app
document.addEventListener('DOMContentLoaded', async () => {
    // Ensure user has a unique ID
    ensureUserId();
    
    // Set up event listeners
    if (messageInput) {
        messageInput.addEventListener('input', handleInputChange);
        messageInput.addEventListener('keydown', handleKeyDown);
    }
    
    if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
    }
    
    // Set up option cards
    document.querySelectorAll('.option-card').forEach(card => {
        card.addEventListener('click', () => {
            const option = card.getAttribute('data-option');
            if (option) {
                selectPath(option);
            }
        });
    });
    
    // Auto-resize textarea
    if (messageInput) {
        messageInput.addEventListener('input', () => {
            messageInput.style.height = 'auto';
            messageInput.style.height = (messageInput.scrollHeight) + 'px';
        });
    }
    
    // Set up sidebar event listeners
    setupSidebarEventListeners();
    
    // Load chat sessions
    await loadChatSessions();
    
    // Don't create a session automatically - only when user clicks on an option card
    // This prevents creating unnecessary sessions on page refresh
});

// Create a new chat session
async function createNewSession() {
    try {
        const userId = getCookie('user_id') || ensureUserId();
        
        const response = await fetch('/api/sessions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: userId,
                name: `Financial Chat ${new Date().toLocaleString()}`
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        currentSessionId = data.session_id;
        console.log("Created new session with ID:", currentSessionId);
        
        // Focus on input
        messageInput.focus();
        
        return data;
    } catch (error) {
        console.error('Error creating new session:', error);
        // Add error message to UI instead of alert
        const errorMessage = document.createElement('div');
        errorMessage.className = 'error-message';
        errorMessage.textContent = 'Failed to create a new session. Please try again.';
        chatContainer.appendChild(errorMessage);
    }
}

// Select a conversation path
async function selectPath(path) {
    // Create a new session only when user clicks on an option card
    if (!currentSessionId) {
        try {
            await createNewSession();
            // Reload sessions to show the new one
            await loadChatSessions();
            // Mark new session as active
            const sessionEl = document.querySelector(`[data-session-id="${currentSessionId}"]`);
            if (sessionEl) {
                sessionEl.classList.add('active');
            }
        } catch (error) {
            console.error('Error creating session:', error);
            return;
        }
    }
    
    // Set the conversation path
    conversationState.path = path;
    conversationState.mode = 'flexible';
    conversationState.inputs = {};
    conversationState.needsPersonalization = false;
    
    // Hide welcome screen and show chat container
    welcomeScreen.style.display = 'none';
    chatContainer.style.display = 'flex';
    
    // Add initial message based on path
    let initialMessage = '';
    let followUpOptions = [];
    
    switch (path) {
        case 'research_company':
            initialMessage = `I'd be happy to help you research a company! You can ask me about any company in several ways:

• **Basic Research**: "Tell me about Apple" or "What do you think of TSLA?"
• **Investment Analysis**: "Should I invest in Microsoft?" or "Is Google a good buy?"
• **Comparison**: "Compare Apple vs Microsoft" or "Which is better: Tesla or Ford?"

What company would you like to explore?`;
            break;
            
        case 'explore_etfs':
            initialMessage = `Great choice! ETFs are an excellent way to build a diversified portfolio. I can help you with:

• **ETF Recommendations**: "What are the best ETFs for beginners?"
• **Portfolio Building**: "Help me create a balanced portfolio"
• **Specific Goals**: "ETFs for retirement" or "Best growth ETFs"
• **Comparison**: "Compare VTI vs VOO" or "International vs domestic ETFs"

What would you like to know about ETFs?`;
            break;
            
        case 'savings_goal':
            initialMessage = `Perfect! I'll help you create a comprehensive savings and investment plan. You can tell me about your goal in any way:

• **Specific Items**: "I want to save for a Hyundai Ioniq 6"
• **General Goals**: "I need to save for a house down payment"
• **Timeline Goals**: "I want to retire in 20 years"
• **Amount Goals**: "I want to save $50,000"

What are you saving for?`;
            break;
    }
    
    addMessageToUI('assistant', initialMessage);
    
    // Clear any input options
    inputOptions.innerHTML = '';
    
    // Focus on input
    messageInput.focus();
}

// Handle input change
function handleInputChange() {
    // Enable/disable send button based on input
    sendButton.disabled = !messageInput.value.trim();
    
    // Auto-resize textarea
    messageInput.style.height = 'auto';
    messageInput.style.height = (messageInput.scrollHeight) + 'px';
}

// Handle key down
function handleKeyDown(e) {
    // Send message on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendButton.disabled && !isProcessing) {
            sendMessage();
        }
    }
}

// Send message
async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message || isProcessing) {
        return;
    }
    
    // Clear input and reset height
    messageInput.value = '';
    messageInput.style.height = 'auto';
    handleInputChange();
    
    // Add user message to UI
    addMessageToUI('user', message);
    
    // Send to server
    await sendToServer(message);
}

// Send a message to the server
async function sendToServer(message) {
    // Show thinking indicator
    const thinkingEl = document.createElement('div');
    thinkingEl.className = 'message assistant';
    thinkingEl.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="thinking">
                Thinking
                <div class="dots">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        </div>
    `;
    messagesContainer.appendChild(thinkingEl);
    scrollToBottom();
    
    // Set processing flag
    isProcessing = true;
    
    try {
        // Send to server with conversation state
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: currentSessionId,
                message: message,
                conversation_state: conversationState
            })
        });
        
        const data = await response.json();
        
        // Remove thinking indicator
        messagesContainer.removeChild(thinkingEl);
        
        // Check if we have separate thinking and final responses
        if (data.thinking_response && data.final_response) {
            // Add thinking response first
            addMessageToUI('assistant_thinking', data.thinking_response);
            
            // Add a small delay before showing final response
            setTimeout(() => {
                addMessageToUI('assistant', data.final_response);
                
                // Check if the response suggests we need more personal information
                if (data.final_response.toLowerCase().includes('personalized') || 
                    data.final_response.toLowerCase().includes('your financial situation') ||
                    data.final_response.toLowerCase().includes('tell me about your')) {
                    
                    // Show option to provide personal details
                    showPersonalizationOption();
                }
            }, 500);
        } else {
            // Fallback to single response (backward compatibility)
            addMessageToUI('assistant', data.response || 'No response received');
            
            // Check if the response suggests we need more personal information
            if (data.response && (data.response.toLowerCase().includes('personalized') || 
                data.response.toLowerCase().includes('your financial situation') ||
                data.response.toLowerCase().includes('tell me about your'))) {
                
                // Show option to provide personal details
                showPersonalizationOption();
            }
        }
        
    } catch (error) {
        console.error('Error sending message:', error);
        
        // Remove thinking indicator
        messagesContainer.removeChild(thinkingEl);
        
        // Add error message
        addMessageToUI('assistant', 'Sorry, there was an error processing your request. Please try again.');
    } finally {
        // Reset processing flag
        isProcessing = false;
        
        // Focus on input
        messageInput.focus();
    }
}

// Show personalization option
function showPersonalizationOption() {
    // Clear previous input options
    inputOptions.innerHTML = '';
    
    const container = document.createElement('div');
    container.className = 'personalization-container';
    
    const message = document.createElement('div');
    message.className = 'personalization-message';
    message.textContent = 'Would you like personalized recommendations based on your financial situation?';
    
    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'personalization-buttons';
    
    const yesButton = document.createElement('button');
    yesButton.className = 'option-button primary';
    yesButton.textContent = 'Yes, personalize my recommendations';
    yesButton.addEventListener('click', () => {
        startPersonalizationFlow();
    });
    
    const noButton = document.createElement('button');
    noButton.className = 'option-button secondary';
    noButton.textContent = 'No, keep it general';
    noButton.addEventListener('click', () => {
        inputOptions.innerHTML = '';
        addMessageToUI('assistant', 'No problem! Feel free to ask me any other questions about your topic.');
    });
    
    buttonContainer.appendChild(yesButton);
    buttonContainer.appendChild(noButton);
    
    container.appendChild(message);
    container.appendChild(buttonContainer);
    
    inputOptions.appendChild(container);
}

// Start personalization flow
function startPersonalizationFlow() {
    // Clear input options
    inputOptions.innerHTML = '';
    
    conversationState.needsPersonalization = true;
    
    addMessageToUI('assistant', `Great! To give you the most relevant advice, I'll need to understand your financial situation. You can share this information naturally in the chat - just tell me about:

💰 **Your income and expenses** - "I make $75k and spend about $4k monthly"
💳 **Current savings and investments** - "I have $20k saved and $50k in my 401k"
🎯 **Your goals and timeline** - "I want to buy a house in 3 years"
⚖️ **Risk tolerance** - "I'm comfortable with moderate risk"

You don't need to share everything at once - just tell me what you're comfortable sharing, and I'll ask for any other details I need.`);
}

// Add message to UI
function addMessageToUI(role, content, timestamp = new Date()) {
    // Hide welcome screen if visible
    if (welcomeScreen.style.display !== 'none') {
        welcomeScreen.style.display = 'none';
        chatContainer.style.display = 'flex';
    }
    
    // Create message element
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;
    
    // Set avatar based on role
    const avatar = role === 'user' ? 'You' : 'AI';
    
    // Format timestamp
    const formattedTime = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    // Format content with markdown
    const formattedContent = formatMessage(content);
    
    // Set HTML
    messageEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            ${formattedContent}
            <div class="message-timestamp">${formattedTime}</div>
        </div>
    `;
    
    // Add to messages container
    messagesContainer.appendChild(messageEl);
    
    // Scroll to bottom
    scrollToBottom();
    
    // Apply syntax highlighting to code blocks if available
    if (typeof hljs !== 'undefined') {
        messageEl.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });
    }
}

// Format message with markdown
function formatMessage(content) {
    try {
        // Check if marked is available
        if (typeof marked !== 'undefined') {
            // Use marked library to convert markdown to HTML
            const html = marked.parse(content);
            return html;
        } else {
            // Fallback if marked is not available
            return content
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code>$1</code>')
                .replace(/^• /gm, '• ')
                .replace(/^💰|💳|🎯|⚖️/gm, '$&');
        }
    } catch (error) {
        console.error("Error formatting message:", error);
        return content;
    }
}

// Scroll to bottom of messages
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Generate and store a unique user ID
function ensureUserId() {
    let userId = getCookie('user_id');
    
    if (!userId) {
        // Generate a random user ID
        userId = 'user_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
        
        // Set cookie that expires in 1 year
        const expiryDate = new Date();
        expiryDate.setFullYear(expiryDate.getFullYear() + 1);
        document.cookie = `user_id=${userId}; expires=${expiryDate.toUTCString()}; path=/; SameSite=Strict`;
    }
    
    return userId;
}

// Get cookie by name
function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null;
}

// Session Management Functions
function setupSidebarEventListeners() {
    // New chat button
    const newChatBtn = document.getElementById('new-chat-btn');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', createNewChatSession);
    }
    
    // Toggle sidebar button
    const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
    if (toggleSidebarBtn) {
        toggleSidebarBtn.addEventListener('click', toggleSidebar);
    }
    
    // Rename chat button
    const renameChatBtn = document.getElementById('rename-chat-btn');
    if (renameChatBtn) {
        renameChatBtn.addEventListener('click', showRenameChatModal);
    }
    
    // Delete chat button
    const deleteChatBtn = document.getElementById('delete-chat-btn');
    if (deleteChatBtn) {
        deleteChatBtn.addEventListener('click', showDeleteChatModal);
    }
    
    // Modal event listeners
    const confirmRenameBtn = document.getElementById('confirm-rename-btn');
    if (confirmRenameBtn) {
        confirmRenameBtn.addEventListener('click', confirmRenameChat);
    }
    
    const confirmDeleteBtn = document.getElementById('confirm-delete-btn');
    if (confirmDeleteBtn) {
        confirmDeleteBtn.addEventListener('click', confirmDeleteChat);
    }
    
    // Enter key in rename input
    const newChatNameInput = document.getElementById('new-chat-name');
    if (newChatNameInput) {
        newChatNameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                confirmRenameChat();
            }
        });
    }
}

// Load chat sessions from server
async function loadChatSessions() {
    try {
        const response = await fetch('/api/sessions');
        const data = await response.json();
        
        const sessionsList = document.getElementById('sessions-list');
        if (!sessionsList) return;
        
        sessionsList.innerHTML = '';
        
        if (data.sessions && data.sessions.length > 0) {
            data.sessions.forEach(session => {
                const sessionEl = createSessionElement(session);
                sessionsList.appendChild(sessionEl);
            });
        } else {
            sessionsList.innerHTML = '<div class="no-sessions">No previous chats</div>';
        }
    } catch (error) {
        console.error('Error loading sessions:', error);
        const sessionsList = document.getElementById('sessions-list');
        if (sessionsList) {
            sessionsList.innerHTML = '<div class="error-sessions">Failed to load chats</div>';
        }
    }
}

// Create session element
function createSessionElement(session) {
    const sessionEl = document.createElement('div');
    sessionEl.className = 'session-item';
    sessionEl.dataset.sessionId = session.session_id;
    
    const date = new Date(session.created_at);
    const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    sessionEl.innerHTML = `
        <div class="session-name">${session.name}</div>
        <div class="session-date">${formattedDate}</div>
        <div class="session-actions">
            <button class="session-action-btn" onclick="renameSession('${session.session_id}')" title="Rename">
                <i class="fas fa-edit"></i>
            </button>
            <button class="session-action-btn" onclick="deleteSession('${session.session_id}')" title="Delete">
                <i class="fas fa-trash"></i>
            </button>
        </div>
    `;
    
    // Add click listener to load session
    sessionEl.addEventListener('click', (e) => {
        // Don't load session if clicking on action buttons
        if (e.target.closest('.session-actions')) return;
        
        loadSession(session.session_id, session.name);
    });
    
    return sessionEl;
}

// Load a specific session
async function loadSession(sessionId, sessionName) {
    try {
        // Update current session
        currentSessionId = sessionId;
        
        // Update UI to show this session as active
        document.querySelectorAll('.session-item').forEach(item => {
            item.classList.remove('active');
        });
        
        const sessionEl = document.querySelector(`[data-session-id="${sessionId}"]`);
        if (sessionEl) {
            sessionEl.classList.add('active');
        }
        
        // Update chat title
        const chatTitle = document.getElementById('current-chat-title');
        if (chatTitle) {
            chatTitle.textContent = sessionName;
        }
        
        // Load chat history
        const response = await fetch(`/api/history/${sessionId}`);
        const data = await response.json();
        
        // Clear current messages
        messagesContainer.innerHTML = '';
        
        // Add messages to UI
        if (data.messages && data.messages.length > 0) {
            data.messages.forEach(msg => {
                if (msg.role !== 'assistant_thinking') { // Skip thinking messages for now
                    addMessageToUI(msg.role, msg.content, new Date(msg.timestamp));
                }
            });
            
            // Show chat container
            welcomeScreen.style.display = 'none';
            chatContainer.style.display = 'flex';
        } else {
            // Show welcome screen if no messages
            welcomeScreen.style.display = 'flex';
            chatContainer.style.display = 'none';
        }
        
        // Reset conversation state
        conversationState = {
            path: null,
            mode: 'flexible',
            inputs: {},
            needsPersonalization: false
        };
        
        // Clear input options
        inputOptions.innerHTML = '';
        
    } catch (error) {
        console.error('Error loading session:', error);
        addMessageToUI('assistant', 'Sorry, there was an error loading this chat session.');
    }
}

// Toggle sidebar visibility
function toggleSidebar() {
    const sidebar = document.getElementById('chat-sidebar');
    const mainArea = document.getElementById('main-chat-area');
    const toggleBtn = document.getElementById('toggle-sidebar-btn');
    const floatingToggle = document.getElementById('sidebar-toggle-floating');
    
    if (sidebar && mainArea) {
        sidebar.classList.toggle('hidden');
        mainArea.classList.toggle('sidebar-hidden');
        
        // Update the main toggle button icon
        if (toggleBtn) {
            const icon = toggleBtn.querySelector('i');
            if (sidebar.classList.contains('hidden')) {
                icon.className = 'fas fa-chevron-right';
                toggleBtn.title = 'Show Sidebar';
            } else {
                icon.className = 'fas fa-chevron-left';
                toggleBtn.title = 'Hide Sidebar';
            }
        }
        
        // Show/hide floating toggle button
        if (floatingToggle) {
            if (sidebar.classList.contains('hidden')) {
                floatingToggle.classList.remove('hidden');
            } else {
                floatingToggle.classList.add('hidden');
            }
        }
    }
}

// Show rename chat modal
function showRenameChatModal() {
    const modal = new bootstrap.Modal(document.getElementById('renameChatModal'));
    const input = document.getElementById('new-chat-name');
    const currentTitle = document.getElementById('current-chat-title').textContent;
    
    if (input) {
        input.value = currentTitle;
        input.select();
    }
    
    modal.show();
}

// Show delete chat modal
function showDeleteChatModal() {
    const modal = new bootstrap.Modal(document.getElementById('deleteChatModal'));
    modal.show();
}

// Confirm rename chat
async function confirmRenameChat() {
    const newName = document.getElementById('new-chat-name').value.trim();
    
    if (!newName || !currentSessionId) return;
    
    try {
        const response = await fetch(`/api/sessions/${currentSessionId}/rename`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name: newName })
        });
        
        if (response.ok) {
            // Update chat title
            const chatTitle = document.getElementById('current-chat-title');
            if (chatTitle) {
                chatTitle.textContent = newName;
            }
            
            // Update session in sidebar
            const sessionEl = document.querySelector(`[data-session-id="${currentSessionId}"]`);
            if (sessionEl) {
                const nameEl = sessionEl.querySelector('.session-name');
                if (nameEl) {
                    nameEl.textContent = newName;
                }
            }
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('renameChatModal'));
            modal.hide();
        }
    } catch (error) {
        console.error('Error renaming session:', error);
    }
}

// Confirm delete chat
async function confirmDeleteChat() {
    if (!currentSessionId) return;
    
    try {
        const response = await fetch(`/api/sessions/${currentSessionId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            // Remove from sidebar
            const sessionEl = document.querySelector(`[data-session-id="${currentSessionId}"]`);
            if (sessionEl) {
                sessionEl.remove();
            }
            
            // Close modal first
            const modal = bootstrap.Modal.getInstance(document.getElementById('deleteChatModal'));
            if (modal) {
                modal.hide();
            }
            
            // Clear current session
            currentSessionId = null;
            
            // Reset UI to welcome screen
            if (messagesContainer) {
                messagesContainer.innerHTML = '';
            }
            
            if (welcomeScreen) {
                welcomeScreen.style.display = 'flex';
            }
            if (chatContainer) {
                chatContainer.style.display = 'none';
            }
            
            if (inputOptions) {
                inputOptions.innerHTML = '';
            }
            
            // Update chat title
            const chatTitle = document.getElementById('current-chat-title');
            if (chatTitle) {
                chatTitle.textContent = 'New Chat';
            }
            
            // Reset conversation state
            conversationState = {
                path: null,
                mode: 'flexible',
                inputs: {},
                needsPersonalization: false
            };
            
            // Remove active state from all sessions
            document.querySelectorAll('.session-item').forEach(item => {
                item.classList.remove('active');
            });
            
            // Create new session after a short delay
            setTimeout(async () => {
                await createNewSession();
                await loadChatSessions();
            }, 100);
            
        } else {
            console.error('Failed to delete session:', response.status);
        }
    } catch (error) {
        console.error('Error deleting session:', error);
    }
}

// Create new chat session
async function createNewChatSession() {
    try {
        await createNewSession();
        
        // Reset conversation state
        conversationState = {
            path: null,
            mode: 'flexible',
            inputs: {},
            needsPersonalization: false
        };
        
        // Clear messages
        if (messagesContainer) {
            messagesContainer.innerHTML = '';
        }
        
        // Show welcome screen
        if (welcomeScreen) {
            welcomeScreen.style.display = 'flex';
        }
        if (chatContainer) {
            chatContainer.style.display = 'none';
        }
        
        // Clear input options
        if (inputOptions) {
            inputOptions.innerHTML = '';
        }
        
        // Update chat title
        const chatTitle = document.getElementById('current-chat-title');
        if (chatTitle) {
            chatTitle.textContent = 'New Chat';
        }
        
        // Reload sessions to show the new one
        await loadChatSessions();
        
        // Mark new session as active
        const sessionEl = document.querySelector(`[data-session-id="${currentSessionId}"]`);
        if (sessionEl) {
            sessionEl.classList.add('active');
        }
        
    } catch (error) {
        console.error('Error creating new session:', error);
    }
}

// Global functions for session actions (called from HTML)
window.renameSession = function(sessionId) {
    // Load the session first, then show rename modal
    const sessionEl = document.querySelector(`[data-session-id="${sessionId}"]`);
    if (sessionEl) {
        const sessionName = sessionEl.querySelector('.session-name').textContent;
        loadSession(sessionId, sessionName).then(() => {
            showRenameChatModal();
        });
    }
};

window.deleteSession = async function(sessionId) {
    try {
        const response = await fetch(`/api/sessions/${sessionId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            // Remove from sidebar immediately
            const sessionEl = document.querySelector(`[data-session-id="${sessionId}"]`);
            if (sessionEl) {
                sessionEl.remove();
            }
            
            // If this was the current session, reset UI
            if (currentSessionId === sessionId) {
                currentSessionId = null;
                
                // Reset UI to welcome screen
                if (messagesContainer) {
                    messagesContainer.innerHTML = '';
                }
                
                if (welcomeScreen) {
                    welcomeScreen.style.display = 'flex';
                }
                if (chatContainer) {
                    chatContainer.style.display = 'none';
                }
                
                if (inputOptions) {
                    inputOptions.innerHTML = '';
                }
                
                // Update chat title
                const chatTitle = document.getElementById('current-chat-title');
                if (chatTitle) {
                    chatTitle.textContent = 'New Chat';
                }
                
                // Reset conversation state
                conversationState = {
                    path: null,
                    mode: 'flexible',
                    inputs: {},
                    needsPersonalization: false
                };
                
                // Create new session after a short delay
                setTimeout(async () => {
                    await createNewSession();
                }, 100);
            }
            
        } else {
            console.error('Failed to delete session:', response.status);
        }
    } catch (error) {
        console.error('Error deleting session:', error);
    }
};
