# OpenPI Knowledge Base

This directory contains the offline knowledge base maintained by the conversation-summarizer agent. The knowledge base serves as a persistent repository of insights, decisions, and learnings extracted from conversations.

## Structure

- **topics/**: Individual knowledge base entries organized by topic
- **categories/**: Entries organized by category (technical, decisions, insights, etc.)
- **index.md**: Master index of all knowledge base entries
- **tags.md**: Tag-based organization for cross-referencing

## File Naming Convention

- Use lowercase with hyphens for separators
- Include creation date: `YYYY-MM-DD-topic-name.md`
- Example: `2024-01-15-data-transforms-architecture.md`

## Categories

- **technical**: Technical discoveries, architectural decisions, code patterns
- **decisions**: Important project decisions and their rationale
- **insights**: Key insights and learnings from discussions
- **processes**: Workflow and process improvements
- **issues**: Problem identification and resolution approaches
- **best-practices**: Established best practices and guidelines

## Usage

The conversation-summarizer agent will automatically:
1. Create new entries for novel topics
2. Update existing entries with new information
3. Maintain cross-references between related topics
4. Keep the index and tags files updated

## Search and Navigation

- Use the index.md file to browse all entries
- Use tags.md to find entries by topic area
- Each entry includes related topics for easy navigation
- Full-text search can be performed across all .md files