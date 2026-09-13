// Service Worker — Chat Gateway PWA
const CACHE = 'gateway-v33';
const PRECACHE = [
  '/chat-gateway/static/manifest.json',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    fetch(e.request)
      .then(resp => {
        if (e.request.url.includes('/static/')) {
          const clone = resp.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return resp;
      })
      .catch(() => caches.match(e.request))
  );
});

// ===== Web Push 通知处理 =====
self.addEventListener('push', e => {
  if (!e.data) return;
  let d = {};
  try { d = e.data.json(); } catch { d = { title: 'Chat Gateway', body: e.data.text() }; }

  const opts = {
    body: d.body || '',
    icon: '/chat-gateway/static/icon-192.png',
    badge: '/chat-gateway/static/icon-192-maskable.png',
    tag: d.tag || 'gateway-event',
    renotify: true,
    data: { url: d.url || '/chat-gateway/admin/' },
    // 振动模式
    vibrate: [200, 100, 200],
  };

  e.waitUntil(self.registration.showNotification(d.title || 'Chat Gateway', opts));
});

// 点击通知跳转
self.addEventListener('notificationclick', e => {
  e.notification.close();
  const url = (e.notification.data && e.notification.data.url) || '/chat-gateway/admin/';
  e.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then(wins => {
      for (const w of wins) {
        if (w.url.includes('chat-gateway') && 'focus' in w) {
          w.navigate(url);
          return w.focus();
        }
      }
      return clients.openWindow(url);
    })
  );
});
