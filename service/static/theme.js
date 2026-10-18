/* Runs before styles to restore the chosen theme without a light/dark flash. */
'use strict';
(() => {
  const root = document.documentElement;
  const key = 'cashlint-theme';
  let saved;
  try { saved = localStorage.getItem(key); } catch { /* Storage may be disabled. */ }
  root.dataset.theme = saved === 'light' || saved === 'dark' ? saved : 'dark';

  function updateButton() {
    const dark = root.dataset.theme === 'dark';
    document.querySelectorAll('.theme-toggle').forEach(button => {
      const action = `Switch to ${dark ? 'light' : 'dark'}`;
      button.title = action;button.setAttribute('aria-label',action);
      button.querySelector('.theme-label').textContent = action;
    });
  }
  document.addEventListener('DOMContentLoaded', () => {
    updateButton();
    document.querySelectorAll('.theme-toggle').forEach(button => {
      button.addEventListener('click', () => {
        const theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
        root.dataset.theme = theme;
        try { localStorage.setItem(key, theme); } catch { /* Theme still works in memory. */ }
        updateButton();
      });
    });
  });
  window.addEventListener('pageshow', () => {
    // Back/forward cache can restore a page from before another page changed it.
    try {
      const theme = localStorage.getItem(key);
      if (theme === 'light' || theme === 'dark') root.dataset.theme = theme;
    } catch { /* Keep the current theme when storage is unavailable. */ }
    updateButton();
  });
})();
