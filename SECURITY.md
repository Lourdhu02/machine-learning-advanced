# Security Policy

## Supported versions

Only the latest commit on `main` is actively supported. Older tagged releases are not patched.

## Reporting a vulnerability

If you find a security issue (e.g. a malicious file in the repo, a dependency with a known CVE, or something that could expose a user's system), **do not open a public issue**.

Email the maintainer directly at **b.lourdhuraju1234@gmail.com** with:

- A description of the vulnerability
- Steps to reproduce (if applicable)
- Any relevant logs or screenshots

You should receive a response within 7 days. If the issue is confirmed, a fix and a public disclosure will follow as soon as practical.

## What qualifies

This is a teaching repository of Python scripts and markdown. The most likely issues are:

- A `requirements.txt` pulling in a package with a known CVE
- A diagram script or `from_scratch.py` doing something unexpected with the local filesystem
- Accidental inclusion of sensitive data (API keys, local paths)

If it doesn't fit the above, email anyway and the maintainer will point you in the right direction.
