# scverse ecosystem registry entry

`meta.yaml` is the entry for [`scverse/ecosystem-packages`](https://github.com/scverse/ecosystem-packages)
(it goes to `packages/scintilla-py/meta.yaml` in a fork of that repository, as part of the listing
pull request). `schema.json` is a verbatim copy of the registry's schema
(`scripts/src/ecosystem_scripts/schema.json`, schema version 2.0), kept here so that
`tests/test_registry_entry.py` can validate the entry on every CI run.

When the registry changes its schema, download the new file over `schema.json`:

```bash
curl -o scverse-registry/schema.json \
  https://raw.githubusercontent.com/scverse/ecosystem-packages/main/scripts/src/ecosystem_scripts/schema.json
```

Do not set `version` (the registry reads it from PyPI). Add `logo: logo.svg` and the file itself if
the project gets a logo.
