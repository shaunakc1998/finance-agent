# Contributing to Finance Agent

Thank you for considering contributing to Finance Agent! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Workflow](#development-workflow)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/finance-agent.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Create a `.env` file with your API keys (see `.env.example`)
7. Run the application: `python web/app.py`

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue tracker to see if the problem has already been reported. If it has and the issue is still open, add a comment to the existing issue instead of opening a new one.

When creating a bug report, include as many details as possible:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear and descriptive title
- A detailed description of the proposed functionality
- Any relevant examples or mockups
- Why this enhancement would be useful to most users

### Pull Requests

1. Create a new branch from `main`: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Commit your changes with a descriptive commit message
5. Push to your fork: `git push origin feature/your-feature-name`
6. Open a pull request against the `main` branch

Pull requests should:

- Have a clear and descriptive title
- Include a detailed description of the changes
- Reference any related issues
- Pass all tests and CI checks
- Be reviewed by at least one maintainer

## Development Workflow

1. Pick an issue to work on or create a new one
2. Create a new branch for your feature or bugfix
3. Write tests for your changes
4. Implement your changes
5. Run tests to ensure they pass
6. Update documentation if necessary
7. Submit a pull request

## Style Guidelines

This project follows PEP 8 style guidelines for Python code. We use flake8 for linting.

- Use 4 spaces for indentation
- Use docstrings for all functions, classes, and modules
- Keep line length to 120 characters or less
- Use meaningful variable and function names
- Write clear comments for complex logic

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests after the first line
- Consider starting the commit message with an applicable emoji:
  - ✨ (sparkles) for new features
  - 🐛 (bug) for bug fixes
  - 📚 (books) for documentation changes
  - ♻️ (recycle) for refactoring
  - 🧪 (test tube) for adding tests
  - 🔧 (wrench) for configuration changes

## License

By contributing to Finance Agent, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
