// Global variables
let currentSessionId = null;
let isProcessing = false;

// DOM elements
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-btn');
const messagesContainer = document.getElementById('messages');
const chatContainer = document.getElementById('chat-container');
const welcomeScreen = document.getElementById('welcome-screen');
const chatHistory = document.getElementById('chat-history');
const newChatButton = document.getElementById('new-chat-btn');
const renameModal = document.getElementById('rename-modal');
const renameInput = document.getElementById('rename-input');
const renameConfirmButton = document.getElementById('rename-confirm-btn');
const closeModalButton = document.querySelector('.close');

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    // Ensure user has a unique ID
    ensureUserId();
    
    // Load chat history
    loadChatSessions();
    
    // Set up event listeners
    messageInput.addEventListener('input', handleInputChange);
    messageInput.addEventListener('keydown', handleKeyDown);
    sendButton.addEventListener('click', sendMessage);
    newChatButton.addEventListener('click', createNewChat);
    closeModalButton.addEventListener('click', closeModal);
    renameConfirmButton.addEventListener('click', confirmRename);
    
    // Set up example prompt cards
    document.querySelectorAll('.prompt-card').forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.getAttribute('data-prompt');
            if (prompt) {
                if (!currentSessionId) {
                    createNewChat().then(() => {
                        messageInput.value = prompt;
                        handleInputChange();
                    });
                } else {
                    messageInput.value = prompt;
                    handleInputChange();
                }
            }
        });
    });
    
    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = (messageInput.scrollHeight) + 'px';
    });
});

// Load chat sessions from the server
async function loadChatSessions() {
    try {
        const response = await fetch('/api/sessions');
        const data = await response.json();
        
        // Clear loading message
        chatHistory.innerHTML = '';
        
        if (data.sessions && data.sessions.length > 0) {
            // Render each chat session
            data.sessions.forEach(session => {
                addChatToSidebar(session.session_id, session.name);
            });
            
            // Load the most recent chat
            loadChat(data.sessions[0].session_id);
        } else {
            // No chats yet, show empty state
            chatHistory.innerHTML = '<div class="no-chats">No chats yet. Start a new conversation!</div>';
        }
    } catch (error) {
        console.error('Error loading chat sessions:', error);
        chatHistory.innerHTML = '<div class="error">Failed to load chat history. Please try refreshing.</div>';
    }
}

// Create a new chat session
async function createNewChat() {
    try {
        const userId = getCookie('user_id') || ensureUserId();
        
        const response = await fetch('/api/sessions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: userId,
                name: `Chat ${new Date().toLocaleString()}`
            })
        });
        
        const data = await response.json();
        currentSessionId = data.session_id;
        
        // Add to sidebar and select it
        addChatToSidebar(data.session_id, data.name, true);
        
        // Clear messages and show empty chat
        messagesContainer.innerHTML = '';
        welcomeScreen.style.display = 'none';
        
        // Focus on input
        messageInput.focus();
        
        return data;
    } catch (error) {
        console.error('Error creating new chat:', error);
        alert('Failed to create a new chat. Please try again.');
    }
}

// Load a specific chat
async function loadChat(sessionId) {
    try {
        currentSessionId = sessionId;
        
        // Update active state in sidebar
        document.querySelectorAll('.chat-item').forEach(item => {
            item.classList.remove('active');
        });
        const chatItem = document.querySelector(`[data-session-id="${sessionId}"]`);
        if (chatItem) {
            chatItem.classList.add('active');
        }
        
        // Get chat history
        const response = await fetch(`/api/history/${sessionId}`);
        const data = await response.json();
        
        // Clear messages
        messagesContainer.innerHTML = '';
        
        // Hide welcome screen if we have messages
        if (data.messages && data.messages.length > 0) {
            welcomeScreen.style.display = 'none';
            
            // Render messages
            data.messages.forEach(message => {
                addMessageToUI(message.role, message.content, new Date(message.timestamp));
            });
            
            // Scroll to bottom
            scrollToBottom();
        } else {
            // Show welcome screen for empty chat
            welcomeScreen.style.display = 'flex';
        }
        
        // Focus on input
        messageInput.focus();
    } catch (error) {
        console.error('Error loading chat:', error);
        alert('Failed to load chat. Please try again.');
    }
}

// Add a chat to the sidebar
function addChatToSidebar(sessionId, name, isActive = false) {
    const chatItem = document.createElement('div');
    chatItem.className = `chat-item ${isActive ? 'active' : ''}`;
    chatItem.setAttribute('data-session-id', sessionId);
    
    chatItem.innerHTML = `
        <div class="chat-item-title">${name}</div>
        <div class="chat-actions">
            <button class="chat-action-btn rename-btn" title="Rename">
                <i class="fas fa-edit"></i>
            </button>
            <button class="chat-action-btn delete-btn" title="Delete">
                <i class="fas fa-trash"></i>
            </button>
        </div>
    `;
    
    // Add click event to load this chat
    chatItem.addEventListener('click', (e) => {
        // Don't trigger if clicking on action buttons
        if (!e.target.closest('.chat-actions')) {
            loadChat(sessionId);
        }
    });
    
    // Add rename button event
    chatItem.querySelector('.rename-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        openRenameModal(sessionId, name);
    });
    
    // Add delete button event
    chatItem.querySelector('.delete-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        deleteChat(sessionId);
    });
    
    // Add to chat history
    if (isActive) {
        // Add to the beginning if active
        chatHistory.insertBefore(chatItem, chatHistory.firstChild);
    } else {
        chatHistory.appendChild(chatItem);
    }
}

// Delete a chat
async function deleteChat(sessionId) {
    if (!confirm('Are you sure you want to delete this chat?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/sessions/${sessionId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            // Remove from sidebar
            const chatItem = document.querySelector(`[data-session-id="${sessionId}"]`);
            if (chatItem) {
                chatItem.remove();
            }
            
            // If we deleted the current chat, load another one or create new
            if (sessionId === currentSessionId) {
                const firstChat = document.querySelector('.chat-item');
                if (firstChat) {
                    loadChat(firstChat.getAttribute('data-session-id'));
                } else {
                    // No chats left, show welcome screen
                    currentSessionId = null;
                    messagesContainer.innerHTML = '';
                    welcomeScreen.style.display = 'flex';
                    chatHistory.innerHTML = '<div class="no-chats">No chats yet. Start a new conversation!</div>';
                }
            }
        } else {
            throw new Error('Failed to delete chat');
        }
    } catch (error) {
        console.error('Error deleting chat:', error);
        alert('Failed to delete chat. Please try again.');
    }
}

// Open rename modal
function openRenameModal(sessionId, currentName) {
    // Set session ID as data attribute
    renameModal.setAttribute('data-session-id', sessionId);
    
    // Set current name in input
    renameInput.value = currentName;
    
    // Show modal
    renameModal.style.display = 'block';
    
    // Focus on input
    renameInput.focus();
}

// Close modal
function closeModal() {
    renameModal.style.display = 'none';
}

// Confirm rename
async function confirmRename() {
    const sessionId = renameModal.getAttribute('data-session-id');
    const newName = renameInput.value.trim();
    
    if (!newName) {
        alert('Please enter a name');
        return;
    }
    
    try {
        const response = await fetch(`/api/sessions/${sessionId}/rename`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: newName
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update name in sidebar
            const chatItem = document.querySelector(`[data-session-id="${sessionId}"]`);
            if (chatItem) {
                chatItem.querySelector('.chat-item-title').textContent = newName;
            }
            
            // Close modal
            closeModal();
        } else {
            throw new Error('Failed to rename chat');
        }
    } catch (error) {
        console.error('Error renaming chat:', error);
        alert('Failed to rename chat. Please try again.');
    }
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
    
    // If no current session, create one
    if (!currentSessionId) {
        const newSession = await createNewChat();
        currentSessionId = newSession.session_id;
    }
    
    // Clear input and reset height
    messageInput.value = '';
    messageInput.style.height = 'auto';
    handleInputChange();
    
    // Add user message to UI
    addMessageToUI('user', message);
    
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
        // Send to server
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: currentSessionId,
                message: message
            })
        });
        
        const data = await response.json();
        
        // Remove thinking indicator
        messagesContainer.removeChild(thinkingEl);
        
        // Add assistant response to UI
        addMessageToUI('assistant', data.response);
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

// Add message to UI
function addMessageToUI(role, content, timestamp = new Date()) {
    // Hide welcome screen if visible
    welcomeScreen.style.display = 'none';
    
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
    
    // Apply syntax highlighting to code blocks
    messageEl.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
    });
}

// Format message with markdown
function formatMessage(content) {
    // Use marked library to convert markdown to HTML
    const html = marked.parse(content);
    
    // Return formatted HTML
    return html;
}

// Scroll to bottom of messages
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Close modal when clicking outside
window.onclick = function(event) {
    if (event.target === renameModal) {
        closeModal();
    }
};

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
