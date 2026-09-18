() => {
  document.body.classList.add('dark');
  // Clicking anywhere in a dropdown's box opens it; clicking outside an open list closes it.
  document.addEventListener('pointerdown', (e) => {
    const box = e.target.closest('.wrap:has(> .wrap-inner)');
    const combo = box && box.querySelector('input[role="combobox"]');
    if (combo && e.target !== combo && !e.target.closest('.options')) {
      e.preventDefault();
      combo.focus();
      combo.click();
    }
    const inOptions = e.target.closest('.options');
    document.querySelectorAll('input[role="combobox"][aria-expanded="true"]').forEach((input) => {
      if (!inOptions && !input.closest('.wrap').contains(e.target)) {
        input.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
        input.blur();
      }
    });
  }, true);
  // Enter sends a chat message (Shift+Enter is ignored for single-line input).
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.target.closest('#chat-input')) {
      e.preventDefault();
      e.stopPropagation();
      const send = document.querySelector('#chat-send');
      if (send && send.offsetParent) send.click();
    }
  }, true);

  // In-app confirmation for every delete. The browser's confirm() popup can be silently disabled
  // ("don't allow this site to prompt you again"), which would make delete buttons do nothing.
  const askConfirm = (title, message, action = 'Delete') => new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'confirm-overlay';
    overlay.innerHTML = '<div class="confirm-box" role="dialog" aria-modal="true"><h3></h3><p></p>' +
      '<div class="confirm-actions"><button type="button" class="confirm-cancel">Cancel</button>' +
      '<button type="button" class="confirm-ok"></button></div></div>';
    overlay.querySelector('h3').textContent = title;
    overlay.querySelector('p').textContent = message;
    overlay.querySelector('.confirm-ok').textContent = action;
    const close = (answer) => { overlay.remove(); document.removeEventListener('keydown', onKey, true); resolve(answer); };
    const onKey = (e) => { if (e.key === 'Escape') close(false); if (e.key === 'Enter') close(true); };
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(false); });
    overlay.querySelector('.confirm-cancel').addEventListener('click', () => close(false));
    overlay.querySelector('.confirm-ok').addEventListener('click', () => close(true));
    document.addEventListener('keydown', onKey, true);
    document.body.appendChild(overlay);
    overlay.querySelector('.confirm-cancel').focus();
  });

  const confirmText = (btn) => {
    if (btn.matches('.chat-del')) {
      const title = btn.closest('.chat-item-row')?.querySelector('.chat-item')?.innerText || 'this chat';
      return ['Delete chat?', `"${title}" will be permanently deleted. This cannot be undone.`];
    }
    if (btn.matches('.row-del') && !btn.closest('.image-models')) {
      const name = btn.closest('.mt-row')?.querySelector('.mt-cell')?.innerText || 'this model';
      return ['Delete model?', `${name} will be removed from this computer. Its files are deleted from disk and cannot be recovered.`];
    }
    if (btn.matches('.g-del')) {
      return ['Delete image?', 'This image will be permanently deleted from this computer.'];
    }
    if (btn.matches('.row-del') && btn.closest('.image-models')) {
      const name = btn.closest('.mt-row')?.querySelector('.mt-cell')?.innerText || 'this image model';
      return ['Delete image model?', `${name} will be removed from this computer. Its files are deleted from disk.`];
    }
    if (btn.matches('.rep-delete')) {
      const report = [...document.querySelectorAll('input[role="combobox"]')].find((i) => i.getAttribute('aria-label') === 'Report')?.value;
      if (!report) return null;
      return ['Delete report?', `${report.trim()} and all of its files will be permanently deleted. This cannot be undone.`];
    }
    if (btn.matches('.dl-del')) {
      return ['Stop download?', `${btn.dataset.ref} will be stopped and removed from downloads. Partially downloaded data is cleaned up the next time Ollama restarts.`, 'Stop download'];
    }
    return null;
  };

  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('button.chat-del, button.row-del, button.rep-delete, button.dl-del, button.g-del');
    if (!btn) return;
    if (btn.dataset.confirmed === '1') { delete btn.dataset.confirmed; return; }  // let the confirmed click through
    e.preventDefault();
    e.stopImmediatePropagation();
    const text = confirmText(btn);
    if (!text || !(await askConfirm(...text))) return;
    if (btn.matches('.g-del')) {
      const box = document.querySelector('#img-del-name textarea, #img-del-name input');
      box.value = btn.dataset.name;
      box.dispatchEvent(new Event('input', {bubbles: true}));
      setTimeout(() => document.querySelector('#img-del-btn').click(), 80);
      return;
    }
    if (btn.matches('.dl-del')) {
      const box = document.querySelector('#dl-remove-ref textarea, #dl-remove-ref input');
      box.value = btn.dataset.ref;
      box.dispatchEvent(new Event('input', {bubbles: true}));
      setTimeout(() => document.querySelector('#dl-remove-btn').click(), 80);
      return;
    }
    btn.dataset.confirmed = '1';
    btn.click();
  }, true);

  // Long messages (yours or the AI's) collapse to a preview with Show more / Show less underneath.
  const COLLAPSE_AT = 420;
  const expanded = new Set();
  const applyCollapse = () => {
    const messages = [...document.querySelectorAll('.chat .message.bot, .chat .message.user')];
    const send = document.querySelector('#chat-send');
    const streaming = !(send && send.offsetParent);  // Stop is shown instead of Send while a reply streams
    messages.forEach((msg, i) => {
      const content = msg.querySelector('.message-content') || msg;
      const live = streaming && i === messages.length - 1;
      let toggle = msg.querySelector(':scope > .collapse-toggle');
      if (live || content.scrollHeight <= COLLAPSE_AT + 80) {
        delete msg.dataset.collapsed;
        if (toggle) toggle.remove();
        return;
      }
      msg.dataset.collapsed = expanded.has(i) ? '0' : '1';
      if (!toggle) {
        toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'collapse-toggle';
        toggle.addEventListener('click', (ev) => {
          ev.stopPropagation();
          expanded.has(i) ? expanded.delete(i) : expanded.add(i);
          applyCollapse();
        });
        msg.appendChild(toggle);
      }
      toggle.textContent = expanded.has(i) ? 'Show less' : 'Show more';
    });
  };
  setInterval(() => { if (document.querySelector('.chat .message')) applyCollapse(); }, 2000);  // large chats can finish rendering late
  let pending = null;
  new MutationObserver((mutations) => {
    const el = (m) => (m.target.nodeType === 1 ? m.target : m.target.parentElement);
    if (mutations.every((m) => el(m) && el(m).closest('.collapse-toggle'))) return;
    clearTimeout(pending);
    pending = setTimeout(applyCollapse, 120);
  }).observe(document.body, {childList: true, subtree: true, characterData: true});
  // Opening a different chat resets which replies are expanded.
  document.addEventListener('click', (e) => { if (e.target.closest('.chat-item, .new-chat')) expanded.clear(); }, true);

  // Chat sidebar collapse / expand (remembered per browser).
  let sidebarState = 'open';
  try { sidebarState = localStorage.getItem('chat-sidebar') || 'open'; } catch (err) { /* storage unavailable */ }
  const applySidebar = () => {
    const card = document.querySelector('.chat-card');
    if (card && card.dataset.sidebar !== sidebarState) card.dataset.sidebar = sidebarState;
  };
  const setSidebar = (state) => {
    sidebarState = state;
    try { localStorage.setItem('chat-sidebar', state); } catch (err) { /* storage unavailable */ }
    applySidebar();
  };
  setInterval(applySidebar, 2000);  // the card may render after this script runs
  applySidebar();
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.side-toggle');
    if (btn) setSidebar(btn.dataset.action === 'collapse' ? 'collapsed' : 'open');
  });

  // + icon shown while the sidebar is collapsed starts a new chat.
  document.addEventListener('click', (e) => {
    if (e.target.closest('.new-chat-icon')) document.querySelector('#chat-new')?.click();
  });

  // x button inside the chat search box clears the search.
  const setupSearchClear = () => {
    const wrap = document.querySelector('.chat-search');
    const field = wrap && wrap.querySelector('input, textarea');
    if (!field) return;
    wrap.classList.toggle('has-text', !!field.value);
    if (wrap.querySelector('.search-clear')) return;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'search-clear';
    btn.title = 'Clear search';
    btn.setAttribute('aria-label', 'Clear search');
    btn.addEventListener('click', () => {
      field.value = '';
      field.dispatchEvent(new Event('input', {bubbles: true}));
      wrap.classList.remove('has-text');
      field.focus();
    });
    (field.parentElement || wrap).appendChild(btn);
    field.addEventListener('input', () => wrap.classList.toggle('has-text', !!field.value));
  };
  setupSearchClear();
  setInterval(setupSearchClear, 2000);

  // Expand icon on a gallery image: full-size view with its prompt and settings.
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.g-expand');
    if (!btn) return;
    const overlay = document.createElement('div');
    overlay.className = 'lightbox';
    overlay.innerHTML = '<button type="button" class="lightbox-close" aria-label="Close"></button>' +
      '<figure><img alt=""><figcaption><div class="lb-prompt"></div><div class="lb-detail"></div></figcaption></figure>';
    overlay.querySelector('img').src = btn.dataset.src;
    overlay.querySelector('.lb-prompt').textContent = btn.dataset.prompt || '';
    overlay.querySelector('.lb-detail').textContent = btn.dataset.detail || '';
    const close = () => { overlay.remove(); document.removeEventListener('keydown', onKey, true); };
    const onKey = (ev) => { if (ev.key === 'Escape') close(); };
    overlay.addEventListener('click', (ev) => { if (!ev.target.closest('figure img')) close(); });
    document.addEventListener('keydown', onKey, true);
    document.body.appendChild(overlay);
  });

  // Attach button tooltip: say whether the current model can take images.
  setInterval(() => {
    const b = document.querySelector('.composer button.attach-btn');
    if (b) b.title = b.classList.contains('docs-only')
      ? "Attach documents and text files (this model can't view images)" : 'Attach images, documents, or text files';
  }, 1500);
}
