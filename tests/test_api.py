from pathlib import Path
import unittest

from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from qdrant_client.models import CollectionStatus

from ragindexer.config import config
from ragindexer.DocumentIndexer import DocumentIndexer
from ragindexer.__main__ import main


class TestIngestWatch(unittest.TestCase):
    def test_qrant_connection(self):
        doc_index = DocumentIndexer()
        info = doc_index.qdrant.info()
        self.assertEqual(info.status, CollectionStatus.GREEN)

    def test_main(self):
        doc_index = DocumentIndexer()
        doc_index.qdrant.empty_collection()

        config.STATE_DB_PATH.unlink(missing_ok=True)

        tot_nb_files = main(only_initial_scan=True)
        self.assertGreaterEqual(tot_nb_files, 0)

        doc_index.qdrant.create_snapshot(Path("tests"))

    def test_embeding(self):
        model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            trust_remote_code=config.EMBEDDING_MODEL_TRUST_REMOTE_CODE,
            backend="torch",
        )
        print(f"Max token length : {model.tokenizer.model_max_length}")

        # Case where the sentences are similar
        fra = "Aujourd'hui, le temps est beau et il pleuvra demain. "
        test_fra = ""
        while len(test_fra) < config.CHUNK_SIZE:
            test_fra += fra
        test_fra = test_fra[: config.CHUNK_SIZE]
        print(f"Sentence length : {len(test_fra)}")
        print(f"Token count : {len(model.tokenizer.tokenize(test_fra))}")

        fra_emb = model.encode([fra], device="cpu", show_progress_bar=False, convert_to_tensor=True)
        print(fra_emb.shape)

        eng = "Today the weather is fine, tomorrow it will rain"
        eng_emb = model.encode([eng], device="cpu", show_progress_bar=False, convert_to_tensor=True)

        self.assertGreater(F.cosine_similarity(fra_emb, eng_emb), 0.9)

        # Case where the sentences are not similar
        fra = "Aujourd'hui, le temps est beau et il pleuvra demain"
        fra_emb = model.encode([fra], device="cpu", show_progress_bar=False, convert_to_tensor=True)

        eng = "Master raven on a perched tree"
        eng_emb = model.encode([eng], device="cpu", show_progress_bar=False, convert_to_tensor=True)

        self.assertLess(F.cosine_similarity(fra_emb, eng_emb), 0.2)

        # Case where the sentences are similar
        fra = "Aujourd'hui, le temps est beau et il pleuvra demain"
        fra_emb = model.encode([fra], device="cpu", show_progress_bar=False, convert_to_tensor=True)

        fra2 = "Maintenant, il y a du soleil, mais demain ce sera un mauvais temps"
        fra2_emb = model.encode(
            [fra2], device="cpu", show_progress_bar=False, convert_to_tensor=True
        )

        self.assertGreater(F.cosine_similarity(fra_emb, fra2_emb), 0, 6)


def process_one_doc(abspath: Path):
    doc_index = DocumentIndexer()
    doc_index.process_file(abspath, force=True)


if __name__ == "__main__":
    a = TestIngestWatch()

    # a.test_qrant_connection()
    # process_one_doc(Path("tests/inputs/docs/Marina Robledo NOTXT.pdf"))
    a.test_main()
    # a.test_embeding()
