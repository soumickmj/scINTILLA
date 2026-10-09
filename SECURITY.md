# Security policy

## Supported versions

Only the latest release on PyPI receives fixes.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private
vulnerability reporting on the
[Security tab](https://github.com/soumickmj/scINTILLA/security/advisories/new)
of this repository. You will get an acknowledgement within a week.

scINTILLA reads files you give it (`.h5ad`, `.h5mu`, CSV, YAML configuration)
and does not open network connections of its own. Configuration files are
parsed with `yaml.safe_load`.
