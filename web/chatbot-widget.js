/**
 * Tarihi Yarimada GraphRAG Chatbot Widget
 *
 * Kullanim:
 *   <script src="chatbot-widget.js"></script>
 *   <script>
 *     TarihiYarimadaChatbot.init({ apiUrl: 'http://localhost:8002' });
 *   </script>
 */

(function() {
    'use strict';

    const DEFAULT_CONFIG = {
        apiUrl: 'http://localhost:8002',
        position: 'bottom-right',
        primaryColor: '#1a365d',
        title: 'Evliya AI',
        placeholder: 'Soru sorun...',
        welcomeMessage: 'Merhaba, ben Evliya! İstanbul Tarihi Yarımada hakkında sorularınızı yanıtlamaya hazırım. Örneğin: "Ayasofya\'yı kim yaptırdı?"'
    };

    let config = {};
    let isOpen = false;
    let isLoading = false;

    // CSS Styles
    const styles = `
        .tyc-widget-container {
            position: fixed;
            z-index: 9999;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .tyc-widget-container.bottom-right {
            bottom: 20px;
            right: 20px;
        }
        .tyc-widget-container.bottom-left {
            bottom: 20px;
            left: 20px;
        }

        .tyc-toggle-btn {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .tyc-toggle-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        }
        .tyc-toggle-btn svg {
            width: 28px;
            height: 28px;
            fill: white;
        }

        .tyc-chat-window {
            position: absolute;
            bottom: 70px;
            right: 0;
            width: 380px;
            height: 500px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
            display: none;
            flex-direction: column;
            overflow: hidden;
        }
        .tyc-chat-window.open {
            display: flex;
        }

        .tyc-header {
            padding: 16px 20px;
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .tyc-header h3 {
            margin: 0;
            font-size: 16px;
            font-weight: 600;
        }
        .tyc-close-btn {
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            padding: 4px;
            opacity: 0.8;
        }
        .tyc-close-btn:hover {
            opacity: 1;
        }

        .tyc-messages {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .tyc-message {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 14px;
            line-height: 1.5;
        }
        .tyc-message.bot {
            background: #f1f5f9;
            color: #1e293b;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }
        .tyc-message.user {
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        .tyc-message.error {
            background: #fee2e2;
            color: #991b1b;
        }

        .tyc-sources {
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid #e2e8f0;
            font-size: 12px;
            color: #64748b;
        }

        .tyc-typing {
            display: flex;
            gap: 4px;
            padding: 12px 16px;
            background: #f1f5f9;
            border-radius: 12px;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }
        .tyc-typing span {
            width: 8px;
            height: 8px;
            background: #94a3b8;
            border-radius: 50%;
            animation: tyc-bounce 1.4s infinite ease-in-out;
        }
        .tyc-typing span:nth-child(1) { animation-delay: -0.32s; }
        .tyc-typing span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes tyc-bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        .tyc-input-area {
            padding: 16px;
            border-top: 1px solid #e2e8f0;
            display: flex;
            gap: 8px;
        }
        .tyc-input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #e2e8f0;
            border-radius: 24px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        .tyc-input:focus {
            border-color: #94a3b8;
        }
        .tyc-send-btn {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: opacity 0.2s;
        }
        .tyc-send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .tyc-send-btn svg {
            width: 20px;
            height: 20px;
            fill: white;
        }
    `;

    // HTML Template
    function createWidget() {
        const container = document.createElement('div');
        container.className = `tyc-widget-container ${config.position}`;
        container.innerHTML = `
            <div class="tyc-chat-window">
                <div class="tyc-header" style="background: ${config.primaryColor}">
                    <h3>${config.title}</h3>
                    <button class="tyc-close-btn" aria-label="Kapat">
                        <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                        </svg>
                    </button>
                </div>
                <div class="tyc-messages"></div>
                <div class="tyc-input-area">
                    <input type="text" class="tyc-input" placeholder="${config.placeholder}">
                    <button class="tyc-send-btn" style="background: ${config.primaryColor}" aria-label="Gonder">
                        <svg viewBox="0 0 24 24">
                            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                        </svg>
                    </button>
                </div>
            </div>
            <button class="tyc-toggle-btn" style="background: ${config.primaryColor}" aria-label="Sohbeti ac">
                <svg viewBox="0 0 24 24">
                    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
                </svg>
            </button>
        `;
        return container;
    }

    // Add styles
    function injectStyles() {
        const styleEl = document.createElement('style');
        styleEl.textContent = styles;
        document.head.appendChild(styleEl);
    }

    // API call
    async function sendQuery(query) {
        const response = await fetch(`${config.apiUrl}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });

        if (!response.ok) {
            throw new Error('API hatasi');
        }

        return response.json();
    }

    // Add message to chat
    function addMessage(text, type) {
        const messagesEl = document.querySelector('.tyc-messages');
        const msgEl = document.createElement('div');
        msgEl.className = `tyc-message ${type}`;

        if (type === 'user') {
            msgEl.style.background = config.primaryColor;
        }

        msgEl.textContent = text;
        messagesEl.appendChild(msgEl);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    // Show/hide typing indicator
    function setTyping(show) {
        const messagesEl = document.querySelector('.tyc-messages');
        const existing = messagesEl.querySelector('.tyc-typing');

        if (show && !existing) {
            const typingEl = document.createElement('div');
            typingEl.className = 'tyc-typing';
            typingEl.innerHTML = '<span></span><span></span><span></span>';
            messagesEl.appendChild(typingEl);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (!show && existing) {
            existing.remove();
        }
    }

    // Handle send
    async function handleSend() {
        const inputEl = document.querySelector('.tyc-input');
        const query = inputEl.value.trim();

        if (!query || isLoading) return;

        inputEl.value = '';
        addMessage(query, 'user');

        isLoading = true;
        setTyping(true);
        document.querySelector('.tyc-send-btn').disabled = true;

        try {
            const result = await sendQuery(query);
            setTyping(false);
            addMessage(result.response, 'bot');
        } catch (error) {
            setTyping(false);
            addMessage('Uzgunum, bir hata olustu. Lutfen tekrar deneyin.', 'error');
        } finally {
            isLoading = false;
            document.querySelector('.tyc-send-btn').disabled = false;
        }
    }

    // Toggle chat window
    function toggleChat() {
        isOpen = !isOpen;
        const windowEl = document.querySelector('.tyc-chat-window');
        windowEl.classList.toggle('open', isOpen);

        if (isOpen) {
            document.querySelector('.tyc-input').focus();
        }
    }

    // Initialize
    function init(userConfig = {}) {
        config = { ...DEFAULT_CONFIG, ...userConfig };

        injectStyles();
        const widget = createWidget();
        document.body.appendChild(widget);

        // Add welcome message
        setTimeout(() => {
            addMessage(config.welcomeMessage, 'bot');
        }, 500);

        // Event listeners
        document.querySelector('.tyc-toggle-btn').addEventListener('click', toggleChat);
        document.querySelector('.tyc-close-btn').addEventListener('click', toggleChat);
        document.querySelector('.tyc-send-btn').addEventListener('click', handleSend);
        document.querySelector('.tyc-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleSend();
        });
    }

    // Expose API
    window.TarihiYarimadaChatbot = { init };
})();
