#! /bin/bash

mkdir -p htmldoc/ragindexer
rm -rf .root tests/*.snapshot
.venv/bin/pytest
.venv/bin/pdoc --html --force --config latex_math=True -o htmldoc ragindexer
.venv/bin/coverage html -d htmldoc/coverage --rcfile tests/coverage.conf
.venv/bin/coverage xml -o htmldoc/coverage/coverage.xml --rcfile tests/coverage.conf
.venv/bin/docstr-coverage src/ragindexer -miP -sp -is -idel --skip-file-doc --badge=htmldoc/ragindexer/doc_badge.svg
.venv/bin/genbadge coverage -l -i htmldoc/coverage/coverage.xml -o htmldoc/ragindexer/cov_badge.svg
mv tests/*.snapshot htmldoc
