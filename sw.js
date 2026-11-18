// Offline support: keep every file the app has loaded (engine, image tools, page) so it
// opens and grades with no connection after the first visit. Bump VERSION to ship updates.
const VERSION = "pregrader-v2";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== VERSION) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  const isPage = e.request.mode === "navigate" || /\.(html|js|zip|webmanifest)$/.test(url.pathname);
  e.respondWith((async () => {
    const cache = await caches.open(VERSION);
    const hit = await cache.match(e.request, { ignoreSearch: true });
    if (hit && !isPage) return hit;                 // big engine files: cache first
    try {
      const res = await fetch(e.request);           // page + code: network first, so updates land
      if (res.ok) cache.put(e.request, res.clone());
      return res;
    } catch (err) {
      if (hit) return hit;
      throw err;
    }
  })());
});
