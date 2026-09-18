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
  // A highlight that slides to the selected item, for the page tabs, sub-tabs and two-way switches
  // (Hugging Face / Ollama, report views), instead of the selection jumping from one item to the next.
  const gliderGroups = () => [
    ...[...document.querySelectorAll('[role="tablist"]')].map((el) => [el, 'button[role="tab"][aria-selected="true"]']),
    ...[...document.querySelectorAll('.segmented .wrap:has(> label)')].map((el) => [el, 'label.selected, label:has(input:checked)']),
  ];
  const placeGliders = () => {
    for (const [group, selector] of gliderGroups()) {
      let glider = group.querySelector(':scope > .glider');
      if (!glider) {
        glider = document.createElement('span');
        glider.className = 'glider';
        glider.setAttribute('aria-hidden', 'true');
        group.prepend(glider);
        group.classList.add('has-glider');
      }
      const item = group.querySelector(selector);
      if (!item || !item.offsetWidth) { glider.style.opacity = '0'; continue; }
      glider.style.opacity = '';
      glider.style.width = `${item.offsetWidth}px`;
      glider.style.height = `${item.offsetHeight}px`;
      glider.style.transform = `translate(${item.offsetLeft}px, ${item.offsetTop}px)`;
      if (!glider.classList.contains('ready')) requestAnimationFrame(() => glider.classList.add('ready'));  // no slide on first paint
    }
  };
  let gliderFrame = 0;
  const queueGliders = () => { cancelAnimationFrame(gliderFrame); gliderFrame = requestAnimationFrame(placeGliders); };
  new MutationObserver(queueGliders).observe(document.body, {subtree: true, attributes: true, attributeFilter: ['class', 'aria-selected']});
  window.addEventListener('resize', queueGliders);
  document.addEventListener('click', () => setTimeout(queueGliders, 0), true);
  setInterval(placeGliders, 1500);  // tabs rendered later (sub-tabs, lazy panels)

  // Hover explanations: anything with data-tip (probes and families in a group's probe list), and every probe in the
  // Probes tab's list (its text comes from the hidden #probe-tips element). One floating panel, so cards never clip it.
  const tipBox = document.createElement('div');
  tipBox.className = 'tip-box';
  tipBox.innerHTML = '<div class="tip-title"></div><div class="tip-body"></div>';
  document.body.appendChild(tipBox);
  let probeTips = null;
  let tipTimer = 0;
  let tipFor = null;
  const tipFrom = (el) => {
    if (el.dataset.tip !== undefined) return [el.dataset.tipTitle || '', el.dataset.tip];
    if (!probeTips) {
      try { probeTips = JSON.parse(document.querySelector('#probe-tips')?.dataset.tips || '{}'); } catch (err) { probeTips = {}; }
    }
    const name = el.innerText.trim().split(/\s+/)[0];  // "dan.DanInTheWild   A subset of..." (probe names have no spaces)
    return probeTips[name] ? [name, probeTips[name]] : null;
  };
  const hideTip = () => { clearTimeout(tipTimer); tipFor = null; tipBox.classList.remove('show'); };
  document.addEventListener('mouseover', (e) => {
    const el = e.target.closest('[data-tip], .probe-list label');
    if (el === tipFor) return;
    hideTip();
    if (!el) return;
    const tip = tipFrom(el);
    if (!tip || !tip[1]) return;
    tipFor = el;
    tipTimer = setTimeout(() => {
      tipBox.querySelector('.tip-title').textContent = tip[0];
      tipBox.querySelector('.tip-body').textContent = tip[1];
      const r = el.getBoundingClientRect();
      tipBox.style.left = '0px';
      tipBox.style.top = '0px';
      tipBox.classList.add('show');
      const w = tipBox.offsetWidth, h = tipBox.offsetHeight;
      const left = Math.min(Math.max(8, r.left), window.innerWidth - w - 8);
      const top = r.bottom + 8 + h < window.innerHeight ? r.bottom + 8 : Math.max(8, r.top - h - 8);
      tipBox.style.left = `${left}px`;
      tipBox.style.top = `${top}px`;
    }, 250);
  });
  document.addEventListener('scroll', hideTip, true);

  // Scrolling an open dropdown list (repository, version, ...) must not scroll the page behind it, even when the
  // list is too short to scroll or has reached its end.
  document.addEventListener('wheel', (e) => {
    const box = e.target.closest('.options');
    if (!box) return;
    const list = box.querySelector('.option-list') || box;
    const atTop = list.scrollTop <= 0;
    const atBottom = list.scrollTop + list.clientHeight >= list.scrollHeight - 1;
    if ((e.deltaY < 0 && atTop) || (e.deltaY > 0 && atBottom)) e.preventDefault();
  }, {passive: false, capture: true});
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
    if (btn.matches('.oc-stop')) {
      return ['Stop Ollama?', 'Loaded models are unloaded, and chats, scans and model downloads that need Ollama will '
        + 'wait until you start it again.', 'Stop Ollama'];
    }
    if (btn.matches('.dl-del')) {
      return ['Delete download?', `${btn.dataset.ref} will be stopped and removed from downloads, and its partially downloaded data deleted. Use pause instead to continue it later.`, 'Delete download'];
    }
    return null;
  };

  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('button.chat-del, button.row-del, button.rep-delete, button.dl-del, button.g-del, button.oc-stop');
    if (!btn) return;
    if (btn.dataset.confirmed === '1') { delete btn.dataset.confirmed; return; }  // let the confirmed click through
    e.preventDefault();
    e.stopImmediatePropagation();
    const finished = btn.matches('.dl-del') && btn.dataset.active === '0';  // just clears a finished row
    const text = finished ? null : confirmText(btn);
    if (!finished && (!text || !(await askConfirm(...text)))) return;
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

  // Pause / resume icon on a download (no confirmation: nothing is lost).
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('button.dl-toggle');
    if (!btn) return;
    const box = document.querySelector('#dl-toggle-ref textarea, #dl-toggle-ref input');
    box.value = btn.dataset.action;
    box.dispatchEvent(new Event('input', {bubbles: true}));
    btn.classList.toggle('pause');
    btn.classList.toggle('resume');  // flip the icon right away; the list refreshes within two seconds
    setTimeout(() => document.querySelector('#dl-toggle-btn').click(), 80);
  });

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
