# GastricPath AI — Security, Privacy & Vulnerability Policy

**Project:** GastricPath AI  
**Security Tier:** Strict Offline / Research Software Isolation

---

## 1. Security Architecture & Offline Guarantee

GastricPath AI is built with an offline-first architecture designed to ensure zero data exfiltration and zero exposure of sensitive research material:
- **Zero External Telemetry:** The application makes zero outbound telemetry calls, analytics tracking, or external advertising pings.
- **Strict Localhost Binding:** The development server binds strictly to `127.0.0.1` (localhost) by default.
- **No Embedded Credentials:** The application requires zero API keys, zero cloud database connections, and zero third-party service tokens to execute inference and research workflows.
- **HTTP Security Headers:**
  - `Content-Security-Policy`: Restricts resource execution to `'self'` and explicitly vetted style CDNs.
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `X-XSS-Protection: 1; mode=block`
  - `Referrer-Policy: strict-origin-when-cross-origin`
  - `Cache-Control: no-cache, no-store, must-revalidate, max-age=0` (Enforcing workstation asset freshness)

---

## 2. Patient & Health Data Privacy Policy

To protect patient privacy in compliance with HIPAA Safe Harbor and GDPR standards:
1. **Zero Protected Health Information (PHI):** No patient names, hospital record numbers, addresses, contact details, or identifying accession numbers are stored or tracked.
2. **Local Database Isolation:** Local runtime databases (`app.db`, `*.db`) and uploaded inference images (`data/uploads/*`) are strictly excluded from version control via `.gitignore`.
3. **Structured Dossier Redaction:** In the Clinical Research & PDF Workbench, missing fields are preserved explicitly as `[Missing / Unspecified]`.

---

## 3. Reporting a Vulnerability

If you discover a security vulnerability or credential exposure risk:
1. Do **not** open a public issue on GitHub.
2. Contact the maintainer directly via email: `devpandey25.in@gmail.com`.
3. Provide detailed steps to reproduce the issue.
4. Maintainers will acknowledge within 48 hours and coordinate a coordinated fix.
