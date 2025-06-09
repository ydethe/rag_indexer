import unittest

from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from ingestwatch.config import config
from ingestwatch.__main__ import main


class TestIngestWatch(unittest.TestCase):
    def test_config(self):
        print(config)

    def test_main(self):
        main()

    def test_embeding(self):
        model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            trust_remote_code=config.EMBEDDING_MODEL_TRUST_REMOTE_CODE,
            backend="openvino",
            model_kwargs={"file_name": "openvino/openvino_model_qint8_quantized.xml"},
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


if __name__ == "__main__":
    a = TestIngestWatch()

    # a.test_config()
    a.test_main()
    # a.test_embeding()
