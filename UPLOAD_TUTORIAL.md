# Upload Tutorial

This directory is the upload target:

```bash
cd /n/home04/yulili/daisuan/dmrg_sacasscf_response_public
```

Do not upload `/n/home04/yulili/daisuan` or the full project directory.
Those contain private chemistry work and cluster output.

## 1. Check The Bundle

Compile Python files:

```bash
find src sharc_interface benchmarks -name '*.py' -print0 | xargs -0 python -m py_compile
```

Run the generic sensitive-term scanner with a private term list that is not
committed:

```bash
printf '%s\n' 'private_keyword_1' 'private_keyword_2' > .private_terms
bash scripts/check_sensitive_terms.sh .private_terms
```

If it prints matches, fix them before upload. Keep `.private_terms` local;
`.gitignore` excludes it.

## 2. Local Git Status

This bundle is already initialized as a local git repository on branch
`main`. Check it before pushing:

```bash
git status --short
git log --oneline --decorate -2
```

Review tracked files:

```bash
git ls-files | sort
```

If you make more edits before upload:

```bash
git diff
git add PATHS_YOU_CHANGED
git status --short
git commit -m "Describe the public change"
```

## 3. Create The Remote Repository

Recommended first remote: a private GitHub repository named something like
`dmrg-sacasscf-response`.

After creating the empty remote repo, push with SSH:

```bash
git branch -M main
git remote add origin git@github.com:YOUR_USER_OR_ORG/dmrg-sacasscf-response.git
git push -u origin main
```

Or push with HTTPS:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USER_OR_ORG/dmrg-sacasscf-response.git
git push -u origin main
```

## 4. Update The Remote Later

After making more public-code changes inside this sanitized directory:

```bash
git status --short
git diff
git add PATHS_YOU_CHANGED
git commit -m "Describe the public change"
git push
```

## 5. Before Making The Repository Public

Do these checks:

```bash
bash scripts/check_sensitive_terms.sh .private_terms
find . -type f \( -name '*.out' -o -name '*.err' -o -name '*.log' -o -name '*.chk' -o -name '*.h5' \)
find src sharc_interface benchmarks -name '*.py' -print0 | xargs -0 python -m py_compile
```

Also decide the license with the group before public release. Until that is
decided, keep the repository private.

## 6. What Not To Commit

Do not commit:

- parent project directories;
- private chemistry directories;
- SHARC trajectory folders;
- cluster scratch paths or node logs;
- checkpoint files, HDF5 files, or binary scratch;
- unpublished private application geometries;
- generated `QM.log`, `QM.err`, `QM.out`, or `PySCF_master.log`.
