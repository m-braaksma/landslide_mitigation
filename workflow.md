# Repository Workflow & Release Process

## Part 1: Making Changes (PR + Branch Workflow)

### Quick Reference
- **Never commit directly to `main`** — the ruleset will reject direct pushes
- Always use a branch → PR → merge workflow

### Step-by-Step: Making a Change

1. **Create a feature branch** (from `main`)
   ```bash
   git checkout main
   git pull origin main          # Stay up to date
   git checkout -b feature/your-feature-name
   ```
   
2. **Make your changes and commit**
   ```bash
   git add .
   git commit -m "Clear commit message describing change"
   ```
   
3. **Push the branch to GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```
   
4. **Open a Pull Request on GitHub**
   - Go to https://github.com/m-braaksma/landslide_mitigation
   - GitHub will prompt you to create a PR
   - Add a description of your changes
   - Review the changes yourself (or have someone review)
   
5. **Merge the PR**
   - Resolve any conversations/comments
   - Click "Squash and merge" (recommended for clean history)
   - Delete the branch after merging
   
6. **Update your local `main`**
   ```bash
   git checkout main
   git pull origin main
   ```

### If You Made a Mistake (Unpushed Commit)

If you committed to `main` locally and can't push:
```bash
git reset HEAD~1              # Undo commit, keep changes
git checkout -b feature/fix-name
git add .
git commit -m "Your message"
git push origin feature/fix-name
# Then open a PR
```


## Part 2: Publishing a New Release to Zenodo

### Before You Release: Testing

1. **Run your full pipeline** to ensure everything works
   ```bash
   # Your test commands here
   ```
   
2. **Update version numbers** if needed (see Version Bumping section below)
   - Edit `CITATION.cff`: Update `version:` field
   - Edit any other version references in docs/code
   
3. **Commit version updates** via a PR (don't commit directly to `main`)

### Creating the Release

1. **Create a git tag** (on `main`)
   ```bash
   git checkout main
   git pull origin main
   git tag -a v0.2.1 -m "Release v0.2.1: Fixed import error in [filename]"
   git push origin v0.2.1
   ```
   
2. **Create a GitHub Release**
   - Go to https://github.com/m-braaksma/landslide_mitigation/releases
   - Click "Draft a new release"
   - Select your tag (e.g., `v0.2.1`)
   - Add release notes describing changes
   - Click "Publish release"
   
3. **Zenodo will auto-detect the release**
   - GitHub → Zenodo webhook will trigger automatically
   - Your DOI will be assigned
   - Check your Zenodo account to verify: https://zenodo.org/account/settings/github/

### After Release

- Update `CITATION.cff` with the new Zenodo DOI (if not already auto-populated)
- Test that your DOI resolves correctly


## Part 3: Version Bumping Guide

### When to Bump?

| Change Type | Bump | Example |
|------------|------|---------|
| Bug fixes, small corrections | Patch (0.2.0 → 0.2.1) | Fixed import error |
| New features, significant changes | Minor (0.2.0 → 0.3.0) | Added new analysis method |
| Major restructuring, breaking changes | Major (0.2.0 → 1.0.0) | Complete rewrite |

### How to Bump

1. Update `CITATION.cff`:
   ```yaml
   version: "0.2.1"
   ```

2. Commit via PR:
   ```bash
   git checkout -b release/v0.2.1
   git add CITATION.cff
   git commit -m "Bump version to v0.2.1"
   git push origin release/v0.2.1
   # Open PR, review, merge
   ```

3. Then follow "Creating the Release" section above.


## Checklists

### Before Every Release
- [ ] Run full pipeline test
- [ ] All changes merged to `main` via PR
- [ ] Version bumped in `CITATION.cff`
- [ ] Release notes written
- [ ] No uncommitted changes locally

### After Every Release
- [ ] GitHub Release created and published
- [ ] Zenodo webhook triggered (check activity)
- [ ] DOI assigned and verified
- [ ] CITATION.cff DOI field updated (if needed)


## Useful Links

- Repository: https://github.com/m-braaksma/landslide_mitigation
- Releases: https://github.com/m-braaksma/landslide_mitigation/releases
- Zenodo: https://zenodo.org/account/settings/github/
- Branch Rules: https://github.com/m-braaksma/landslide_mitigation/rules

