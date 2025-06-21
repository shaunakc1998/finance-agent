# GitHub Setup Guide

This guide will walk you through the process of setting up your finance-agent project on GitHub with protected main branch, code analysis, and open source licensing.

## Prerequisites

- [Git](https://git-scm.com/downloads) installed on your local machine
- A [GitHub](https://github.com/) account
- Your finance-agent project code ready to be pushed

## Step 1: Create a New GitHub Repository

1. Log in to your GitHub account
2. Click on the "+" icon in the top right corner and select "New repository"
3. Name your repository (e.g., "finance-agent")
4. Add a description (optional)
5. Choose "Public" for an open source project
6. Do NOT initialize the repository with a README, .gitignore, or license (we'll push these from your local project)
7. Click "Create repository"

## Step 2: Initialize Git in Your Local Project

If your project is not already a Git repository, initialize it:

```bash
cd /path/to/finance-agent
git init
```

## Step 3: Add Your Files to Git

```bash
# Add all files
git add .

# Commit the files
git commit -m "Initial commit"
```

## Step 4: Connect Your Local Repository to GitHub

Replace `yourusername` with your GitHub username:

```bash
git remote add origin https://github.com/yourusername/finance-agent.git
```

## Step 5: Push Your Code to GitHub

```bash
# Push to the main branch
git push -u origin main
```

## Step 6: Verify GitHub Actions Workflows

After pushing your code, GitHub will automatically run the workflows you've set up:

1. Go to your repository on GitHub
2. Click on the "Actions" tab
3. You should see the workflows running or completed

## Step 7: Set Up Branch Protection

The branch-protection.yml workflow should automatically set up branch protection for the main branch. To verify:

1. Go to your repository on GitHub
2. Click on "Settings"
3. Click on "Branches" in the left sidebar
4. Under "Branch protection rules", you should see a rule for the main branch

If the branch protection is not set up automatically, you can manually set it up:

1. Go to your repository on GitHub
2. Click on "Settings"
3. Click on "Branches" in the left sidebar
4. Click on "Add rule"
5. Enter "main" as the branch name pattern
6. Check the following options:
   - Require pull request reviews before merging
   - Require status checks to pass before merging
   - Require branches to be up to date before merging
   - Include administrators
7. Click "Create"

## Step 8: Set Up GitHub Secrets

For the GitHub Actions workflows to work properly, you need to set up secrets for your API keys:

1. Go to your repository on GitHub
2. Click on "Settings"
3. Click on "Secrets and variables" in the left sidebar
4. Click on "Actions"
5. Click on "New repository secret"
6. Add the following secrets:
   - OPENAI_API_KEY
   - ALPHA_VANTAGE_API_KEY
7. Click "Add secret" for each

## Step 9: Enable GitHub Pages (Optional)

If you want to host documentation for your project:

1. Go to your repository on GitHub
2. Click on "Settings"
3. Scroll down to "GitHub Pages"
4. Select "main" as the source branch and "/docs" as the folder
5. Click "Save"

## Step 10: Create a Release (Optional)

To create a release for your project:

1. Go to your repository on GitHub
2. Click on "Releases" on the right sidebar
3. Click on "Create a new release"
4. Enter a tag version (e.g., "v1.0.0")
5. Enter a release title
6. Add release notes
7. Click "Publish release"

## Congratulations!

Your finance-agent project is now set up on GitHub with protected main branch, code analysis, and open source licensing. You can now start accepting contributions from the community!
