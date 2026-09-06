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
      if (result.reflection) {
        const reflNote = document.createElement('div');
        reflNote.className = 'agent-tool-reflection';
        reflNote.style.marginTop = '6px';
        reflNote.style.padding = '6px 8px';
        reflNote.style.borderRadius = '4px';
        reflNote.style.fontSize = '0.82em';
        const isSuccess = result.outcome === 'success';
        reflNote.style.background = isSuccess ? 'rgba(46, 204, 113, 0.15)' : 'rgba(231, 76, 60, 0.15)';
        reflNote.style.border = isSuccess ? '1px solid rgba(46, 204, 113, 0.4)' : '1px solid rgba(231, 76, 60, 0.4)';
        reflNote.innerHTML = `<strong>Outcome:</strong> ${(result.outcome || '').toUpperCase()} | <strong>Distance:</strong> ${(result.distance_travelled_m || 0).toFixed(3)}m | <strong>Max Height:</strong> ${(result.max_height_reached_m || 0).toFixed(3)}m<br><strong>Feedback:</strong> ${result.reflection}`;
        card.appendChild(reflNote);
      }
    } else {
      card.classList.add('tool-refused');
      status.textContent = `✗ ${result.reason || result.error || 'refused'}: ${result.detail || ''}`;
    }
  }

  showThinkingIndicator() {
    this.removeThinkingIndicator();
    if (!this.chatLog) return;
    const el = document.createElement('div');
    el.id = 'agentThinkingCard';
    el.className = 'agent-msg agent-msg-assistant agent-thinking-card';
    el.innerHTML = `
      <div class="agent-msg-header">🤖 GLM Agent</div>
      <div class="agent-thinking-body">
        <span class="thinking-spinner"></span>
        <span class="thinking-label">GLM is thinking...</span>
      </div>
    `;
    this.chatLog.appendChild(el);
    this.chatLog.scrollTop = this.chatLog.scrollHeight;
  }

  removeThinkingIndicator() {
    const el = document.getElementById('agentThinkingCard');
    if (el) el.remove();
  }

  appendThoughtBlock() {
    if (!this.chatLog) return null;
    const container = document.createElement('details');
    container.className = 'agent-thought-container';
    container.open = true;

    const summary = document.createElement('summary');
    summary.className = 'agent-thought-summary';
    summary.innerHTML = '🧠 <em>Thinking process...</em>';

    const content = document.createElement('div');
    content.className = 'agent-thought-content';

    container.appendChild(summary);
    container.appendChild(content);
    this.chatLog.appendChild(container);
    this.chatLog.scrollTop = this.chatLog.scrollHeight;
    return content;
  }

  cancelStream() {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    this.removeThinkingIndicator();
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
    this.showThinkingIndicator();

    this.abortController = new AbortController();
    const target = this.targetGetter();
    const confirm = this.confirmTokenGetter();

    let thoughtContentDiv = null;
    let textContentDiv = null;

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
        this.removeThinkingIndicator();
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
            this.handleAgentEvent(
              ev,
              (chunk) => {
                if (!thoughtContentDiv) {
                  thoughtContentDiv = this.appendThoughtBlock();
                }
                thoughtContentDiv.textContent += chunk;
                if (this.chatLog) this.chatLog.scrollTop = this.chatLog.scrollHeight;
              },
              (chunk) => {
                if (!textContentDiv) {
                  textContentDiv = this.appendMessage('assistant', '');
                }
                textContentDiv.textContent += chunk;
                if (this.chatLog) this.chatLog.scrollTop = this.chatLog.scrollHeight;
              }
            );
          } catch (e) {
            console.warn('Failed to parse SSE line', e, jsonStr);
          }
        }
      }

      this.removeThinkingIndicator();
      this.history.push({ role: 'user', content: text });
      const finalReply = (textContentDiv && textContentDiv.textContent) || '';
      if (finalReply) {
        this.history.push({ role: 'assistant', content: finalReply });
      }

    } catch (err) {
      this.removeThinkingIndicator();
      if (err.name !== 'AbortError') {
        this.appendMessage('system', `Network or runtime failure: ${err.message}`);
      }
    } finally {
      this.removeThinkingIndicator();
      this.setStreaming(false);
      this.onActionExecuted();
    }
  }

  handleAgentEvent(ev, onThought, onText) {
    switch (ev.type) {
      case 'thought':
        this.removeThinkingIndicator();
        if (onThought && ev.content) {
          onThought(ev.content);
        }
        break;

      case 'text':
        this.removeThinkingIndicator();
        if (onText && ev.content) {
          onText(ev.content);
        }
        break;

      case 'tool_call':
        this.removeThinkingIndicator();
        this.appendToolCard(ev.name, ev.args, ev.id);
        break;

      case 'tool_result':
        this.updateToolResult(ev.id, ev.result);

        // Reflect movement directly on the 3D Viewer & Sequence Timeline!
        if (ev.result && ev.result.ok && this.viewer) {
          // Sync robot 3D scene pose if reported by tool
          if (ev.result.pose && typeof this.viewer.setRobotPosition === 'function') {
            const p = ev.result.pose;
            const yawRad = (p.yaw !== undefined ? p.yaw : 0) * (Math.PI / 180);
            this.viewer.setRobotPosition(p.x, p.z, yawRad);
          }

          // Handle course loading
          if (ev.name === 'bittle_load_course' && ev.result.preset && typeof this.viewer.loadCourse === 'function') {
            this.viewer.loadCourse(ev.result.preset);
          } else if (ev.name === 'bittle_reset_pose' && typeof this.viewer.resetRobotPosition === 'function') {
            this.viewer.resetRobotPosition();
          }

          if (ev.result.moveset && ev.result.moveset.frames) {
            const seqName = ev.result.name || ev.result.expression || ev.result.moveset.name || 'Sequence';
            if (typeof window.loadAndPlayMoveset === 'function') {
              window.loadAndPlayMoveset(ev.result.moveset);
            } else {
              this.viewer.playSequence(ev.result.moveset.frames, {
                name: seqName,
                description: ev.result.moveset.description || '',
                loop: false
              });
            }
          } else if (ev.name === 'bittle_do_skill' && ev.result.skill) {
            const skill = ev.result.skill;
            if (!this.viewer.playSequence(skill, { name: skill, loop: false })) {
              if (skill === 'sit') {
                this.viewer.setPose({ 0: 0, 8: -30, 9: -30, 10: 80, 11: 80, 12: 40, 13: 40, 14: 75, 15: 75 });
              } else if (skill === 'balance' || skill === 'up') {
                this.viewer.resetPose('stand');
              } else if (skill === 'rest') {
                this.viewer.resetPose('rest');
                this.viewer.setEstop(false);
              }
            }
          } else if (ev.name === 'bittle_move_joints' && ev.result.applied) {
            this.viewer.setPose(ev.result.applied);
          } else if (ev.name === 'bittle_estop') {
            this.viewer.setEstop(true);
          }

          if (ev.name === 'bittle_save_moveset' && typeof window.refreshMovesetLibrary === 'function') {
            window.refreshMovesetLibrary();
          }
        }
        break;

      case 'done':
        this.removeThinkingIndicator();
        if (ev.final_message && onText) {
          onText(ev.final_message);
        }
        break;

      case 'error':
        this.removeThinkingIndicator();
        const errText = ev.content || ev.detail || ev.error || 'Unknown agent error';
        this.appendMessage('system', `[Error] ${errText}`);
        break;
    }
  }
}
