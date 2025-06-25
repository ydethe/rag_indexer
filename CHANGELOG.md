# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## Unreleased

<small>[Compare with latest](https://github.com/ydethe/ragindexer/compare/94eb7c1b309263c638f3e92cea40ebdc13d83ac1...HEAD)</small>

### Added

- Added publication of qdrant snapshot on github pages ([8c637a3](https://github.com/ydethe/ragindexer/commit/8c637a3d757a238a35840a2a49619879a21e856c) by Yann de The).
- Added link to doc pages ([080304e](https://github.com/ydethe/ragindexer/commit/080304ec4a90cbc50efd03be87d3ec716991a381) by Yann de The).
- Added github pages publication ([3fb26fb](https://github.com/ydethe/ragindexer/commit/3fb26fb240d07ec595caa6a4a9f297ea93514dc8) by Yann de The).
- Added MIN_EXPECTED_CHAR env var ([c5011fc](https://github.com/ydethe/ragindexer/commit/c5011fc759b4aaddadb40d7e2135e5e2a11f438c) by Yann de The).
- Added test documents ([9dc3019](https://github.com/ydethe/ragindexer/commit/9dc30199f450f5879a9a42cfcf65613d380b8344) by Yann de The).
- Added logging msg to tell if OCR cache is reused ([168a97d](https://github.com/ydethe/ragindexer/commit/168a97d020fb14ecb668bf2cbf0b6828f34c6e59) by Yann de The).
- Added emails watcher ([9de852c](https://github.com/ydethe/ragindexer/commit/9de852c33850ac3076924e2bf2abe370cdb609fc) by Yann de The).
- Added emails observer ([511da4a](https://github.com/ydethe/ragindexer/commit/511da4ad52cce5a7c21ebe0b26cc961658565be2) by Yann de The).
- Added API keys and remove of deprecated method QdrantClient.search ([12d727f](https://github.com/ydethe/ragindexer/commit/12d727f8c96d018c1ca1d803222664a7ca528fca) by Yann de The).
- Added page progress ([fd78e72](https://github.com/ydethe/ragindexer/commit/fd78e7266a6019c9934b22e044b614420da79e9c) by Yann de The).
- Added new_ocr for later integration ([c56d9d7](https://github.com/ydethe/ragindexer/commit/c56d9d7bd1b2784733176d15be352b4bcc1c22f3) by Yann de The).
- Added a cache for OCR ([c4ce7d9](https://github.com/ydethe/ragindexer/commit/c4ce7d92e45fb17da590060795df5e44c3d8b59e) by Yann de The).
- Added OCR cache ([8dc4205](https://github.com/ydethe/ragindexer/commit/8dc42053c8f3363d779c84b6c648386b2c6cc273) by Yann de The).
- Added unit test to check embedding relevance ([d9e072f](https://github.com/ydethe/ragindexer/commit/d9e072ffc96e832b2ca336a983f2ee1127f5460a) by Yann de The).
- Added progress bars for file analysis. pdf tested OK ([be49834](https://github.com/ydethe/ragindexer/commit/be49834155c7b30d3ad5a4abfd08174d3cd229fe) by Yann de The).
- Added sqlite db to keep track of the indexed files ([8f5c67f](https://github.com/ydethe/ragindexer/commit/8f5c67ffa1c27698802ac9d3b1274b087656d5d9) by Yann de The).

### Fixed

- Fixed Action ([85cd5bb](https://github.com/ydethe/ragindexer/commit/85cd5bbb0c96c318d57cbac3fd021dec516d339b) by Yann de The).
- Fixed Exception raised when no text read ([88c078f](https://github.com/ydethe/ragindexer/commit/88c078f4f708ef90ec31a8169858d13bef75bd77) by Yann de The).
- Fixed empty chunks ([a72e49a](https://github.com/ydethe/ragindexer/commit/a72e49a4ea0873e01336aaf7bc2f98999fd3475c) by Yann de The).
- Fixed pid computation ([e3840c1](https://github.com/ydethe/ragindexer/commit/e3840c16f5f9a94623924f6bcbb730f3850e9faf) by Yann de The).
- Fixed OCR cache determination ([91dbc4d](https://github.com/ydethe/ragindexer/commit/91dbc4dfe0972f2483e0b48f4ea10dfc14c42706) by Yann de The).
- Fixed qdrant hostname ([def96ad](https://github.com/ydethe/ragindexer/commit/def96ad7d8ce11dc8d62e911e6c7cd68251388a4) by Yann de The).
- Fixed NLTK downloads ([10e1dae](https://github.com/ydethe/ragindexer/commit/10e1daef293c042512a0b4b43514aadb9dbfff4a) by Yann de The).
- Fixed bug where a whole pdf file is skipped if one page has no text ([301e5eb](https://github.com/ydethe/ragindexer/commit/301e5eb7db159b4ca5a69fce8c26c5fc21224ed9) by Yann de The).
- Fixed image name ([eadcf30](https://github.com/ydethe/ragindexer/commit/eadcf3044c877e6b9cd01b20ab3a3bb7204e509f) by Yann de The).
- Fixed pdf error ([cff80bb](https://github.com/ydethe/ragindexer/commit/cff80bbb69e929a6e53d1617ef30b091e88336d8) by Yann de The).
- Fixed docker compose stack ([814adf6](https://github.com/ydethe/ragindexer/commit/814adf6ccf826eabe9e3004d762c2115f89d2e4c) by Yann de The).
- Fixing pipeline ([00107b2](https://github.com/ydethe/ragindexer/commit/00107b29eaad9b53df18e2037222e1a938494966) by Yann de The).

### Removed

- Removed openvino backend ([173f987](https://github.com/ydethe/ragindexer/commit/173f987437949e0c7802bdc0aefa9e078f8f104a) by Yann de The).
- Removed frontend ([ae2d107](https://github.com/ydethe/ragindexer/commit/ae2d1079aada5871a07e4478d6e560d5eae88884) by Yann de The).
- Removed cuda whl files ([da171ab](https://github.com/ydethe/ragindexer/commit/da171ab5d7d811ff9dde4ccfb4bf06e055efef3a) by Yann de The).
- Removed openapi key from ingestion image ([a56abcb](https://github.com/ydethe/ragindexer/commit/a56abcb933960ce1a4a4cea974bc23ec51b37607) by Yann de The).

<!-- insertion marker -->
