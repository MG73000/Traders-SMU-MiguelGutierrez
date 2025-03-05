# Learning Path Repository - Traders@SMU Alpha Program

Welcome to your learning journey! This repository is structured to guide you through a progressive learning path with organized materials, exercises, and projects.

## Welcome

This repository serves as your central hub for all course materials. Each level is broken down into weekly modules, allowing you to progress at a steady pace while building on previous knowledge.

## Repository Structure

This repository contains materials organized by levels (L1, L2) and weeks, along with a final project.

## Structure

- **L1**: Level 1 materials (Junior)
  - Week1 through Week8
- **L2**: Level 2 materials (Associate)
  - Week1 through Week8
- **Final_Project**: Final project materials

## Getting Started

Choose your level and navigate to the corresponding week to find materials and instructions.

## Step 1: Create Your Own Copy of the Repository

1. Go to the [Traders@SMU student-template repository](https://github.com/Traders-SMU/student-template)
2. Click the green "Use this template" button near the top of the page
3. Select "Create a new repository"
4. Name your repository (e.g., "alpha-program-yourname")
5. Choose whether to make it public or private
6. Click "Create repository from template"

## Step 2: Clone the Repository to Your Computer

### Using GitHub Desktop (Recommended for beginners):

1. Open GitHub Desktop
2. Click "File" → "Clone repository"
3. Select your newly created repository
4. Choose where to save it on your computer
5. Click "Clone"

### Using Git Command Line:
```bash
# Navigate to where you want to store the repository
cd Documents/Projects

# Clone your repository (replace USERNAME and REPO-NAME with your information)
git clone https://github.com/USERNAME/REPO-NAME.git

# Move into the repository folder
cd REPO-NAME
```

## Step 3: Find Your Level and Week

1. Open the repository folder on your computer
2. Navigate to either the L1 (Junior) or L2 (Associate) folder, depending on your program level
3. Find the folder for the current week (Week1, Week2, etc.)
4. Read the README.md file in that folder for instructions on the week's assignment

## Step 4: Complete Your Assignment

1. Edit the files as instructed in the weekly README
2. For Week 1, fill out the Personal Goals Document with your information
3. For subsequent weeks, complete the coding assignments and exercises

## Step 5: Submit Your Assignment

### Using GitHub Desktop:

1. Open GitHub Desktop
2. You'll see a list of changed files
3. Enter a summary of your changes (e.g., "Completed Week 1 assignment")
4. Click "Commit to main"
5. Click "Push origin" to upload your changes

### Using Git Command Line:
```bash
# Create a branch for your submission (replace X with your level and Y with the week)
git checkout -b LX-WY-submission

# Add all your changes
git add .

# Commit your changes with a message
git commit -m "Completed Week Y assignment"

# Push your changes to GitHub
git push origin LX-WY-submission
```

## Step 6: Create a Pull Request

1. Go to your repository on GitHub
2. You should see a message about your recently pushed branch
3. Click "Compare & pull request"
4. Add any comments about your submission
5. Click "Create pull request"

## Receiving Updates Throughout the Course

As the course progresses, we'll add new materials for upcoming weeks (Week 2, Week 3, etc.). You'll need to update your repository to get these new materials. Here's how:

### Option 1: Using GitHub Web Interface (Easiest Method)

1. Navigate to your repository on GitHub
2. Look for the message that says "This branch is X commits behind Traders-SMU:main"
3. Click on "Sync fork" near the top of the page
4. Click "Update branch" to get the latest changes
5. Once updated online, pull the changes to your local repository:
   - In GitHub Desktop: Click "Fetch origin" then "Pull origin"
   - In Git command line: Run `git pull origin main`

### Option 2: Using Git Command Line (More Control)

First time setup (only do this once):
```bash
# Add the original repository as an "upstream" remote
git remote add upstream https://github.com/Traders-SMU/student-template.git
```

Whenever you need to get updates:
```bash
# Make sure you're on your main branch
git checkout main

# Get the latest changes from the original repository
git fetch upstream

# Merge those changes into your local repository
git merge upstream/main

# Push the updated content to your GitHub repository
git push origin main
```

### Handling Merge Conflicts

If you've modified files that were also updated in the original repository, you might see a "merge conflict." This happens when Git can't automatically determine which changes to keep.

If you encounter merge conflicts:

1. Look for files marked as conflicted in GitHub Desktop or listed in the command line output
2. Open these files in a text editor - you'll see sections marked with `<<<<<<<`, `=======`, and `>>>>>>>`
3. Edit the file to keep the changes you want (remove the marker lines and unwanted content)
4. Save the file
5. Mark the conflict as resolved:
   - In GitHub Desktop: Right-click the file and select "Mark as resolved"
   - In Git command line: Run `git add <filename>` for each resolved file
6. Complete the merge:
   - In GitHub Desktop: Click "Commit merge"
   - In Git command line: Run `git commit`

If you're unsure how to resolve conflicts, reach out to your instructor for assistance.

### When to Update Your Repository

We recommend checking for updates:
- At the beginning of each week
- Whenever an announcement is made about new materials
- If you encounter references to files that don't exist in your repository

## Common Issues and Solutions

### "Fatal: Repository not found" Error

This usually means the repository URL is incorrect or you don't have access.
- Check that you're using the correct URL
- Make sure you've created your own copy of the template repository

### "Permission denied" Error

This typically means you don't have write access to the repository.
- Make sure you're pushing to your own repository, not the original template
- Check that you've set up your GitHub authentication correctly

### Changes Not Showing Up

If your changes aren't appearing on GitHub:
- Make sure you've committed your changes (`git commit`)
- Make sure you've pushed your changes (`git push`)
- Check that you're on the correct branch

### "Updates were rejected" Error

This usually means there are changes on GitHub that you don't have locally.
- Pull the latest changes first: `git pull origin main`
- Then try pushing again

## Getting Help

If you encounter any issues:
- Ask a classmate who might be familiar with Git
- Contact your program instructor
- Search for your error message on [Stack Overflow](https://stackoverflow.com/)
- Refer to the [GitHub documentation](https://docs.github.com/en)

## Git/GitHub Terminology

- **Repository (Repo)**: A storage location for your project
- **Clone**: Creating a local copy of a repository on your computer
- **Commit**: Saving your changes to the local repository
- **Push**: Uploading your committed changes to the remote repository (GitHub)
- **Pull**: Downloading changes from the remote repository to your local copy
- **Branch**: A separate version of the repository for development
- **Pull Request (PR)**: A request to merge changes from one branch into another
- **Fork**: A copy of a repository that you manage
- **Upstream**: The original repository that you created your copy from
- **Merge**: Combining changes from different sources
- **Merge Conflict**: When Git can't automatically combine changes

Remember, everyone starts somewhere with Git! Don't be afraid to ask questions if you get stuck.

```bash
# After downloading or creating the new files
git add L1/Week2/
git commit -m "Add Week 2 materials"
git push origin main
```