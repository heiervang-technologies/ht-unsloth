## HT Fork Management

This is an [HT (Heiervang Technologies)](https://github.com/heiervang-technologies) fork. For full details on branch conventions, sync workflow, and contribution process, see the [Fork Management Guide](https://github.com/orgs/heiervang-technologies/discussions/3).

### Branch Conventions

- **`main`** — Clean fast-forward mirror of upstream. Never commit directly.
- **`ht`** — Default branch with HT-specific changes. All PRs target `ht`.
- **Feature branches** — Create from `ht`, squash-merge back via PR.

### Sync Workflow

1. Fast-forward `main` from upstream: `git fetch upstream && git merge --ff-only upstream/main`
2. Rebase `ht` onto updated `main`: `git checkout ht && git rebase main`
3. Force-push: `git push --force-with-lease origin ht`

### Commit Standards

- Conventional commits: `feat(<scope>): description`, `fix(<scope>): description`
- Linear history on `ht` — no merge commits, squash fix-up commits
- One commit per logical change

For all questions and inquiries about this fork, please use the [HT Discussions](https://github.com/orgs/heiervang-technologies/discussions) page.

---

# 🦥 Contributing to Unsloth

Thank you for not only using Unsloth but also for being interested in helping out! We value all contributions, whether they come in the form of code, ideas, support for others or just by simply spreading the word of Unsloth! 💕

- **[Support the Community](https://github.com/unslothai/unsloth/issues)**: Answer questions, review pull requests, or assist others in discussions.
- **Fix Bugs**: Identify and resolve issues with the existing codebase.
- **Submit Ideas**: Request new features or share enhancements you'd like to see.
- **Develop Features**: Implement new functionality or improve existing tools which can be done via PRs.
- **[Improve Documentation](https://docs.unsloth.ai/)**: Help by creating guides, FAQs, or enhancing clarity.

One of the best ways to support us is by spreading the word about Unsloth! Share how it’s powering your amazing projects in blog posts or social media, and inspire others to explore its potential. Even a simple star on our repo goes a long way in showing your support and helping the community grow. 🌟

## Submitting Issues
If you find a bug or have a feature idea, we’d love to hear from you! Here’s how to make your submission stand out:

### Reporting Bugs
1. **Search First**: Check if the issue has already been reported using GitHub’s search bar under Issues.
2. **Details Matter**: Is this on Google Colab, Kaggle, or on another platform service? Are you using Unsloth's official notebook? Include your OS, Python version, and other relevant details. For bugs, a concise code snippet that reproduces the issue is incredibly helpful.
3. **Be Thorough**: Attach screenshots, traceback logs, or any additional information that might speed up resolution.

## Spread the Word
Your support extends beyond code:
- Spread the word by writing about Unsloth in blogs or social media.
- Share how Unsloth powers your projects.
- Star our repository to show your appreciation.

Finally, please be mindful of our [Code of Conduct](https://github.com/unslothai/unsloth/blob/main/CODE_OF_CONDUCT.md) to ensure a welcoming and inclusive environment for everyone.

Thank you so much for reading and we hope you have lots of fun using Unsloth! 🦥
