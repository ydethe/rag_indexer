from pathlib import Path
import re
import time

import markdown
import gradio as gr
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from openai.types.responses import Response

from . import logger
from .config import config
from .QdrantIndexer import QdrantIndexer


class ChatDocFontend(object):
    def __init__(self):
        # --- Configuration ---
        self.openai = OpenAI(
            # This is the default and can be omitted
            api_key=config.OPENAI_API_KEY,
        )

        # --- Initialisation ---
        self.encoder = SentenceTransformer(
            config.EMBEDDING_MODEL, trust_remote_code=config.EMBEDDING_MODEL_TRUST_REMOTE_CODE
        )
        vector_size = self.encoder.get_sentence_embedding_dimension()
        self.qdrant = QdrantIndexer(vector_size=vector_size)

    def link_citations(self, text):
        """
        Remplace [1], [2]... par des liens HTML cliquables vers les ancres correspondantes.
        """

        def replacer(match):
            num = match.group(1)
            return f'<a href="#src{num}" style="text-decoration:none;">[{num}]</a>'

        return re.sub(r"\[(\d+)\]", replacer, text)

    def rag_with_anchored_sources(self, query):
        query_vector = self.encoder.encode(query).tolist()

        t0 = time.time()
        hits = self.qdrant.search(query_vector=query_vector, limit=config.QDRANT_QUERY_LIMIT)
        elapsed = time.time() - t0
        logger.info(f"Vector search completed in {elapsed:.1f} s")

        context_chunks = []
        sources_seen = {}
        html_sources = ""

        for i, hit in enumerate(hits):
            text = hit.payload.get("text", "")
            source_file = Path(hit.payload.get("source", "source inconnue"))
            source_name = source_file.parts[-1]
            num = len(sources_seen) + 1

            context_chunks.append(f"[{num}] {text}")

            # Si on n'a pas encore affiché ce fichier
            if source_file not in sources_seen:
                sources_seen[source_file] = num
                if source_file.suffix in (".pdf", ".docx", ".xlsx", ".xlsm", ".md", ".txt"):
                    html_sources += f"""
                    <div id="src{num}" style='margin-top:20px; padding:10px; border:1px solid #ccc;'>
                        <b>[{num}] {source_file}</b><br>
                        <iframe src="{config.DAV_ROOT}/{source_name}" width="100%" height="300px"></iframe>
                    </div>
                    """
                else:
                    logger.warning(f"Source not handled '{source_file}'")

        context = "\n\n".join(context_chunks)

        prompt = f"""
        Tu es un assistant intelligent. Voici des extraits de documents numérotés. Utilise-les pour répondre précisément à la question. Cite les extraits utilisés avec leur numéro, comme ceci : [1].

        Contexte :
        {context}

        Question : {query}

        Réponse (avec citations) :
        """

        t0 = time.time()
        response: Response = self.openai.responses.create(
            input=prompt,
            model=config.OPEN_MODEL_PREF,
            temperature=0.3,
        )
        elapsed = time.time() - t0
        logger.info(f"LLM response got in {elapsed:.1f} s")

        html_answer = markdown.markdown(response.output_text)
        answer = self.link_citations(html_answer)

        full_output = f"""
        <div style='font-family:Arial, sans-serif; line-height:1.5;'>{answer}</div>
        <hr>
        <h4>Sources utilisées :</h4>
        {html_sources}
        """

        return full_output

    def start(self):
        # --- Gradio Interface ---
        iface = gr.Interface(
            fn=self.rag_with_anchored_sources,
            inputs=gr.Textbox(label="Votre question"),
            outputs=gr.HTML(label="Réponse + sources"),
            title="Johncloud - ChatDoc",
            flagging_mode="never",
        )

        iface.launch(server_name="0.0.0.0")
