# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Finance Agent seriously. If you believe you've found a security vulnerability, please follow these steps:

1. **Do not disclose the vulnerability publicly**
2. **Email us at [security@example.com](mailto:security@example.com)** with details about the vulnerability
3. Include the following information:
   - Type of vulnerability
   - Full path to the vulnerable file
   - Proof of concept or steps to reproduce
   - Potential impact

## What to Expect

- We will acknowledge receipt of your vulnerability report within 48 hours
- We will provide a more detailed response within 7 days, indicating the next steps in handling your report
- We will keep you informed of our progress towards resolving the issue
- After the issue is resolved, we will publicly acknowledge your responsible disclosure (unless you prefer to remain anonymous)

## Security Measures

This project implements several security measures:

1. **Automated Security Scanning**: We use GitHub's CodeQL analysis to automatically scan for common vulnerabilities
2. **Dependency Monitoring**: We use Dependabot to keep dependencies up-to-date
3. **Pre-commit Hooks**: We use pre-commit hooks to check for security issues before code is committed
4. **Code Review**: All pull requests are reviewed for security issues before being merged

## Security Best Practices for Contributors

When contributing to this project, please follow these security best practices:

1. Keep all dependencies up-to-date
2. Do not commit sensitive information (API keys, passwords, etc.)
3. Follow the principle of least privilege
4. Validate all user inputs
5. Use parameterized queries for database operations
6. Follow secure coding practices
