# Cloudflare protection for the public research demo

Configured through the owner's authenticated Cloudflare dashboard on 2026-10-19. Both rules are active and explicitly scoped to `http.host eq "neurosteer.keqichen.top"`; other hostnames in the zone are not matched.

## Scanner probes

Custom rule: **NeuroSteer - block scanner probes**, action **Block**.

```text
(http.host eq "neurosteer.keqichen.top" and (
  http.request.uri.path contains "/.env" or
  http.request.uri.path contains "/.git" or
  starts_with(http.request.uri.path, "/wp-") or
  http.request.uri.path eq "/xmlrpc.php" or
  starts_with(lower(http.request.uri.path), "/phpmyadmin") or
  http.request.method in {"TRACE" "CONNECT"} or
  lower(http.user_agent) contains "sqlmap" or
  lower(http.user_agent) contains "masscan"
))
```

## Page/reconnect rate limit

Rate rule: **NeuroSteer - page and reconnect rate limit**.

```text
(http.host eq "neurosteer.keqichen.top" and
 http.request.uri.path in {"/" "/_stcore/stream"})
```

Count by IP; threshold 30 matching requests per 10 seconds; block for 10 seconds. Static resources and the health endpoint are excluded. This limits page refreshes and WebSocket handshakes, not messages sent inside an already established WebSocket; it is not a per-user inference quota. No paid plan was enabled, no zone-wide Bot Fight/Under Attack switch was changed, and no authentication gate was added.

## Evidence

- Normal homepage and health requests returned 200.
- `/.env`, `/wp-login.php`, and a homepage request with a `sqlmap-test` user agent returned 403.
- A bounded 45-request check returned 32 HTTP 200 responses and 13 HTTP 429 responses. This verifies enforcement, not a precise performance/abuse benchmark.
- The updated bilingual notice and real generation were verified in Streamlit's test runner against the live backend. Public-browser regression verification is recorded in the maintenance PR.

To adjust or disable: Cloudflare dashboard → `keqichen.top` → Security → Security rules, then find the two rule names above. Keep the hostname condition when editing them. To increase public usage capacity, revisit limits and application-level inference queuing rather than assuming WebSocket messages are counted by these edge HTTP rules.
