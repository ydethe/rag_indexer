import gradio as gr
from sentence_transformers import SentenceTransformer
import openai
import os
import re

from .config import config
from .QdrantIndexer import QdrantIndexer


class ChatDocFontend(object):
    def __init__(self):
        # --- Configuration ---
        openai.api_key = config.OPENAI_API_KEY
        self.base_url = "http://localhost:8000/static/docs/"

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

        hits = self.qdrant.search(query_vector=query_vector, limit=config.QDRANT_QUERY_LIMIT)

        context_chunks = []
        sources_seen = {}
        html_sources = ""

        for i, hit in enumerate(hits):
            text = hit.payload.get("text", "")
            source_file = hit.payload.get("source", "source inconnue")
            source_name = os.path.basename(source_file)
            num = len(sources_seen) + 1

            context_chunks.append(f"[{num}] {text}")

            # Si on n'a pas encore affiché ce fichier
            if source_file not in sources_seen:
                sources_seen[source_file] = num
                if source_file.endswith(".pdf"):
                    html_sources += f"""
                    <div id="src{num}" style='margin-top:20px; padding:10px; border:1px solid #ccc;'>
                        <b>[{num}] {source_name}</b><br>
                        <iframe src="{self.base_url}{source_name}" width="100%" height="300px"></iframe>
                    </div>
                    """
                else:
                    html_sources += f"""
                    <div id="src{num}" style='margin-top:20px;'>
                        <b>[{num}]</b> <a href="{self.base_url}{source_name}" target="_blank">{source_name}</a>
                    </div>
                    """

        context = "\n\n".join(context_chunks)

        prompt = f"""
    Tu es un assistant intelligent. Voici des extraits de documents numérotés. Utilise-les pour répondre précisément à la question. Cite les extraits utilisés avec leur numéro, comme ceci : [1].

    Contexte :
    {context}

    Question : {query}

    Réponse (avec citations) :
    """

        response = openai.ChatCompletion.create(
            model=config.OPEN_MODEL_PREF,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        raw_answer = response["choices"][0]["message"]["content"]
        answer = self.link_citations(raw_answer)

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
            title="RAG avec citations cliquables et aperçus de documents",
        )

        iface.launch()
