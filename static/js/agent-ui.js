/**
 * GLM Agent UI & Harness Client
 * Manages chat interactions, streaming thoughts, tool execution cards,
 * and synchronizing agent movements with the 3D digital twin.
 */

export class AgentUI {
  constructor(options = {}) {
    this.viewer = options.viewer;
    this.chatLog = document.getElementById('agentChatLog');
    this.promptInput = document.getElementById('agentPromptInput');
    this.sendBtn = document.getElementById('agentSendBtn');
    this.cancelBtn = document.getElementById('agentCancelBtn');
    this.autoExecToggle = document.getElementById('agentAutoExecToggle');
    this.targetGetter = options.getTarget || (() => 'sim');
    this.confirmTokenGetter = options.getConfirmToken || (() => null);
    this.onActionExecuted = options.onActionExecuted || (() => {});

    this.history = [];
    this.isStreaming = false;
    this.abortController = null;

    this.initEvents();
  }

  initEvents() {
    if (this.sendBtn) {
      this.sendBtn.onclick = () => this.sendPrompt();
    }
    if (this.promptInput) {
      this.promptInput.onkeydown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.sendPrompt();
        }
      };
    }
    if (this.cancelBtn) {
      this.cancelBtn.onclick = () => this.cancelStream();
    }

    // Example chip prompts
    document.querySelectorAll('.agent-prompt-chip').forEach(chip => {
      chip.onclick = () => {
        if (this.promptInput) {
          this.promptInput.value = chip.dataset.prompt || chip.textContent.trim();
          this.sendPrompt();
        }
      };
    });
  }

  appendMessage(role, text) {
    if (!this.chatLog) return null;
    const msgDiv = document.createElement('div');
    msgDiv.className = `agent-msg agent-msg-${role}`;
    
    const roleSpan = document.createElement('div');
    roleSpan.className = 'agent-msg-header';
    roleSpan.textContent = role === 'user' ? '👤 Operator' : role === 'assistant' ? '🤖 GLM Agent' : '⚠️ System';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'agent-msg-content';
    contentDiv.textContent = text;

    msgDiv.appendChild(roleSpan);
    msgDiv.appendChild(contentDiv);
    this.chatLog.appendChild(msgDiv);
    this.chatLog.scrollTop = this.chatLog.scrollHeight;
    return contentDiv;
  }

  appendToolCard(toolName, args, id) {
    if (!this.chatLog) return null;
    const card = document.createElement('div');
    card.className = 'agent-tool-card';
    card.id = `tool-${id}`;

    const title = document.createElement('div');
    title.className = 'agent-tool-title';
    title.innerHTML = `⚙️ <strong>${toolName}</strong>`;

    const argsPre = document.createElement('pre');
    argsPre.className = 'agent-tool-args';
    argsPre.textContent = JSON.stringify(args, null, 2);

    const status = document.createElement('div');
    status.className = 'agent-tool-status';
    status.textContent = 'executing...';

    card.appendChild(title);
    card.appendChild(argsPre);
    card.appendChild(status);

    this.chatLog.appendChild(card);
    this.chatLog.scrollTop = this.chatLog.scrollHeight;
    return card;
  }

  updateToolResult(id, result) {
    const card = document.getElementById(`tool-${id}`);
    if (!card) return;

    const status = card.querySelector('.agent-tool-status');
    if (result.ok) {
      card.classList.add('tool-success');
      status.textContent = `✓ ${result.response || 'success'}`;
      if (result.clamped && result.adjustments) {
        const clampNote = document.createElement('div');
        clampNote.className = 'agent-tool-clamp';
        clampNote.textContent = `Clamped: ${JSON.stringify(result.adjustments)}`;
        card.appendChild(clampNote);
      }
    } else {
      card.classList.add('tool-refused');
      status.textContent = `✗ ${result.reason || result.error || 'refused'}: ${result.detail || ''}`;
    }
  }

  cancelStream() {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    this.setStreaming(false);
    this.appendMessage('system', 'Agent run cancelled by operator.');
  }

  setStreaming(streaming) {
    this.isStreaming = streaming;
    if (this.sendBtn) this.sendBtn.disabled = streaming;
    if (this.cancelBtn) this.cancelBtn.style.display = streaming ? 'inline-block' : 'none';
  }

  async sendPrompt() {
    if (!this.promptInput) return;
    const text = this.promptInput.value.trim();
    if (!text || this.isStreaming) return;

    this.promptInput.value = '';
    this.appendMessage('user', text);
    this.setStreaming(true);

    this.abortController = new AbortController();
    const target = this.targetGetter();
    const confirm = this.confirmTokenGetter();

    let thoughtContentDiv = null;

    try {
      const response = await fetch('/api/agent/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: text,
          history: this.history,
          target: target,
          confirm: confirm,
          stream: true
        }),
        signal: this.abortController ? this.abortController.signal : undefined
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        this.appendMessage('system', `Error ${response.status}: ${err.detail || 'Agent service error'}`);
        this.setStreaming(false);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop(); // keep remainder

        for (const block of lines) {
          if (!block.startsWith('data: ')) continue;
          const jsonStr = block.slice(6).trim();
          if (!jsonStr) continue;

          try {
            const ev = JSON.parse(jsonStr);
            this.handleAgentEvent(ev, (text) => {
              if (!thoughtContentDiv) {
                thoughtContentDiv = this.appendMessage('assistant', '');
              }
              thoughtContentDiv.textContent += text;
            });
          } catch (e) {
            console.warn('Failed to parse SSE line', e, jsonStr);
          }
        }
      }

      this.history.push({ role: 'user', content: text });
      if (thoughtContentDiv && thoughtContentDiv.textContent) {
        this.history.push({ role: 'assistant', content: thoughtContentDiv.textContent });
      }

    } catch (err) {
      if (err.name !== 'AbortError') {
        this.appendMessage('system', `Network or runtime failure: ${err.message}`);
      }
    } finally {
      this.setStreaming(false);
      this.onActionExecuted();
    }
  }

  handleAgentEvent(ev, appendThought) {
    switch (ev.type) {
      case 'thought':
        appendThought(ev.content);
        break;

      case 'tool_call':
        this.appendToolCard(ev.name, ev.args, ev.id);
        break;

      case 'tool_result':
        this.updateToolResult(ev.id, ev.result);

        // Reflect movement directly on the 3D Viewer!
        if (ev.result && ev.result.ok && this.viewer) {
          if (ev.name === 'bittle_move_joints' && ev.result.applied) {
            this.viewer.setPose(ev.result.applied);
          } else if (ev.name === 'bittle_do_skill') {
            if (ev.result.skill === 'sit') {
              this.viewer.setPose({ 0: 0, 8: -30, 9: -30, 10: 80, 11: 80, 12: 40, 13: 40, 14: 75, 15: 75 });
            } else if (ev.result.skill === 'balance' || ev.result.skill === 'up') {
              this.viewer.resetPose();
            } else if (ev.result.skill === 'rest') {
              this.viewer.resetPose();
              this.viewer.setEstop(false);
            }
          } else if (ev.name === 'bittle_estop') {
            this.viewer.setEstop(true);
          }
        }
        break;

      case 'done':
        if (ev.final_message && !appendThought.hasContent) {
          appendThought(ev.final_message);
        }
        break;

      case 'error':
        this.appendMessage('system', `[${ev.error}] ${ev.detail || ''}`);
        break;
    }
  }
}
