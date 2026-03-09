# Contributing to Moonlab

Welcome to the Moonlab contributor community! This guide covers everything you need to start contributing.

## Quick Start

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/moonlab.git
cd moonlab

# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)

# Test
make test

# Create branch
git checkout -b feature/your-feature
```

## Ways to Contribute

### Code Contributions

- **Bug fixes**: Check [open issues](https://github.com/tsotchke/moonlab/issues?q=is%3Aopen+is%3Aissue+label%3Abug)
- **New features**: Discuss in [Discussions](https://github.com/tsotchke/moonlab/discussions) first
- **Performance improvements**: Profile-driven optimizations welcome
- **Algorithm implementations**: New quantum algorithms

### Non-Code Contributions

- **Documentation**: Fix typos, improve explanations, add examples
- **Bug reports**: Detailed reports with reproducible examples
- **Feature requests**: Well-reasoned proposals with use cases
- **Community support**: Help answer questions in Discussions

## Contribution Process

### 1. Find or Create an Issue

- Check [existing issues](https://github.com/tsotchke/moonlab/issues)
- For new features, open a discussion first
- Comment on issues you want to work on

### 2. Fork and Branch

```bash
# Fork via GitHub UI, then:
git clone https://github.com/YOUR_USERNAME/moonlab.git
cd moonlab
git remote add upstream https://github.com/tsotchke/moonlab.git
git checkout -b feature/your-feature
```

### 3. Develop

- Follow the [code style guide](code-style.md)
- Write tests for new functionality
- Update documentation as needed

### 4. Test

```bash
# Run all tests
make test

# Run specific test suite
./bin/test_quantum_gates

# Run with sanitizers (debug build)
./bin/test_quantum_gates 2>&1
```

### 5. Submit Pull Request

- Fill out the PR template completely
- Reference related issues
- Ensure CI passes
- Request review

## Pull Request Guidelines

### PR Title Format

```
<type>: <short description>

Types:
- feat: New feature
- fix: Bug fix
- perf: Performance improvement
- refactor: Code refactoring
- docs: Documentation only
- test: Test additions/fixes
- chore: Build/tooling changes
```

### PR Description Template

```markdown
## Summary
Brief description of changes.

## Motivation
Why is this change needed?

## Changes
- List of specific changes
- Another change

## Testing
How was this tested?

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)
```

### Review Process

1. **Automated checks**: CI must pass
2. **Code review**: At least one maintainer approval
3. **Testing**: Reviewer may test locally
4. **Merge**: Maintainer merges after approval

## Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). In summary:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Prioritize community well-being

## Development Resources

### Guides

| Guide | Description |
|-------|-------------|
| [Development Setup](development-setup.md) | Environment configuration |
| [Code Style](code-style.md) | Formatting and conventions |
| [Testing Guide](testing-guide.md) | How to write and run tests |

### Reference

| Document | Description |
|----------|-------------|
| [Architecture](../architecture/index.md) | System design |
| [API Reference](../api/index.md) | Function documentation |
| [Error Codes](../reference/error-codes.md) | Error catalog |

## Good First Issues

Look for issues labeled [`good-first-issue`](https://github.com/tsotchke/moonlab/labels/good-first-issue). These are:

- Well-defined in scope
- Have clear acceptance criteria
- Don't require deep codebase knowledge
- Often include guidance

## Getting Help

### Communication Channels

- **GitHub Discussions**: General questions, feature discussions
- **GitHub Issues**: Bug reports, specific tasks
- **Pull Request Comments**: Code-specific questions

### Response Times

- **Issues**: Usually within 48 hours
- **PRs**: Initial review within 1 week
- **Discussions**: Community-driven, varies

## Recognition

Contributors are recognized in:

- [CONTRIBUTORS.md](https://github.com/tsotchke/moonlab/blob/main/CONTRIBUTORS.md)
- Release notes for significant contributions
- Special thanks in documentation for major efforts

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](https://github.com/tsotchke/moonlab/blob/main/LICENSE).

## Questions?

If something isn't covered here, please:

1. Check [FAQ](../faq.md)
2. Search existing [Discussions](https://github.com/tsotchke/moonlab/discussions)
3. Open a new discussion if needed

Thank you for contributing to Moonlab!

