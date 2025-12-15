import os
import pickle
import sys

import faiss
import gradio as gr
import numpy as np
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

bedrock_embedder = BedrockEmbeddings()


def count_tokens(text):
    return len(text.split())


def load_and_chunk_text(filepath, max_tokens=350, overlap_tokens=40):
    """
    Chunk text by tokens, not characters. Default: 350 tokens per chunk, 40 token overlap.
    Adjust max_tokens for your model's context window (Nova Lite is limited).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin1") as f:
            text = f.read()
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    chunk = []
    current_tokens = 0
    current_chapter = "Unknown Source"
    chapter_prefixes = ["CHAPTER ", "Chapter "]
    for para in paragraphs:
        if any(para.startswith(prefix) for prefix in chapter_prefixes):
            current_chapter = para
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens > max_tokens and chunk:
            chunk_text = " ".join(chunk)
            chunks.append({"text": chunk_text, "source": current_chapter})
            if overlap_tokens > 0:
                overlap_text = " ".join(chunk)[-overlap_tokens:]
                overlap_words = chunk_text.split()[-overlap_tokens:]
                chunk = [" ".join(overlap_words)] if overlap_words else []
                current_tokens = count_tokens(" ".join(chunk))
            else:
                chunk = []
                current_tokens = 0
        chunk.append(para)
        current_tokens += para_tokens
    if chunk:
        chunk_text = " ".join(chunk)
        chunks.append({"text": chunk_text, "source": current_chapter})
    return chunks


def build_faiss_index(chunks, embedder):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_documents(texts)
    embeddings = np.array(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings, chunks


def save_faiss_and_chunks(index, embeddings, chunks, base_path):
    faiss.write_index(index, base_path + ".index")
    np.save(base_path + "_embeddings.npy", embeddings)
    with open(base_path + "_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_faiss_and_chunks(base_path):
    index = faiss.read_index(base_path + ".index")
    embeddings = np.load(base_path + "_embeddings.npy")
    with open(base_path + "_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, embeddings, chunks


def retrieve_context(query, embedder, index, chunks, top_k=3):
    query_emb = embedder.embed_query(query)
    query_emb = np.array([query_emb], dtype=np.float32)
    D, I = index.search(query_emb, top_k)
    results = []
    for idx in I[0]:
        chunk = chunks[idx]
        results.append(f"[Source: {chunk['source']}]:\n{chunk['text']}")
    return "\n---\n".join(results)


def build_pdo_prompt(persona, directive, output):
    """Builds a PDO prompt string from components."""
    return f"Persona: {persona}\nDirective: {directive}\nOutput: {output}"


def main():
    print("Loading or building FAISS index and knowledge base chunks...")
    kb_path = os.path.join(os.path.dirname(__file__), "source", "harrypotter.txt")
    rag_dir = os.path.join(os.path.dirname(__file__), "rag_index")
    os.makedirs(rag_dir, exist_ok=True)
    faiss_base = os.path.join(rag_dir, "faiss_harrypotter")
    embedder = bedrock_embedder
    if (
        os.path.exists(faiss_base + ".index")
        and os.path.exists(faiss_base + "_embeddings.npy")
        and os.path.exists(faiss_base + "_chunks.pkl")
    ):
        index, embeddings, chunk_list = load_faiss_and_chunks(faiss_base)
    else:
        chunks = load_and_chunk_text(kb_path)
        index, embeddings, chunk_list = build_faiss_index(chunks, embedder)
        save_faiss_and_chunks(index, embeddings, chunk_list, faiss_base)

    persona = "You are an expert Harry Potter assistant, knowledgeable and friendly."
    directive = (
        "Answer user questions using the provided RAG context. "
        "If the answer is not in the context, state clearly that you don't have the information. "
        "Only provide an imaginative answer if the user agrees, and ensure it is thoughtful and appropriate."
    )
    output = "Be concise, cite the source if possible, and use a friendly tone."

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", type=str, help="Persona for the chatbot")
    parser.add_argument("--directive", type=str, help="Directive for the chatbot")
    parser.add_argument("--output", type=str, help="Output style for the chatbot")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    args, unknown = parser.parse_known_args()
    if args.persona:
        persona = args.persona
    if args.directive:
        directive = args.directive
    if args.output:
        output = args.output

    llm_model = "amazon.nova-lite-v1:0"
    llm_client = ChatBedrockConverse(model=llm_model)

    def get_chain(p, d, o):
        sys_prompt = build_pdo_prompt(p, d, o)
        prompt = ChatPromptTemplate(
            [
                ("system", sys_prompt),
                (MessagesPlaceholder(variable_name="chat_history")),
                ("ai", "Relevant context from knowledge base:\n{context}"),
                ("human", "{query}"),
            ]
        )
        return prompt | llm_client

    def gradio_to_flat_history(history, latest_user_msg=None, latest_bot_msg=None):
        flat = []
        for msg in history:
            if msg["role"] == "user":
                flat.append(("human", msg["content"]))
            elif msg["role"] == "assistant":
                flat.append(("ai", msg["content"]))
        if latest_user_msg:
            flat.append(("human", latest_user_msg))
        if latest_bot_msg:
            flat.append(("ai", latest_bot_msg))
        return flat

    def flat_to_gradio_history(flat):
        messages = []
        for role, msg in flat:
            if role == "human":
                messages.append({"role": "user", "content": msg})
            elif role == "ai":
                messages.append({"role": "assistant", "content": msg})
        return messages

    def chat_fn(
        message, history, persona_val=None, directive_val=None, output_val=None
    ):
        p = persona_val if persona_val is not None else persona
        d = directive_val if directive_val is not None else directive
        o = output_val if output_val is not None else output
        chain = get_chain(p, d, o)
        chat_history = gradio_to_flat_history(history, latest_user_msg=message)
        rag_context = retrieve_context(message, embedder, index, chunk_list)
        response = chain.invoke(
            {"query": message, "chat_history": chat_history, "context": rag_context}
        )
        chat_history.append(("ai", response.content))
        return response.content, flat_to_gradio_history(chat_history)

    if args.cli:
        print(
            "Welcome to Harry PRT Chat Bot! Type 'exit' or 'quit' to end the conversation."
        )
        print(
            f"Current Persona: {persona}\nCurrent Directive: {directive}\nCurrent Output: {output}"
        )
        print("Type '/pdo' to edit Persona, Directive, Output.")
        chat_history = []
        p, d, o = persona, directive, output
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat. Goodbye!")
                break
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting chat. Goodbye!")
                break
            if user_input.strip() == "/pdo":
                print(
                    f"Current Persona: {p}\nCurrent Directive: {d}\nCurrent Output: {o}"
                )
                p = input("Edit Persona (or press Enter to keep): ") or p
                d = input("Edit Directive (or press Enter to keep): ") or d
                o = input("Edit Output (or press Enter to keep): ") or o
                print("PDO prompt updated.")
                continue
            if not user_input:
                continue
            chain = get_chain(p, d, o)
            rag_context = retrieve_context(user_input, embedder, index, chunk_list)
            chat_history.append(("human", user_input))
            response = chain.invoke(
                {
                    "query": user_input,
                    "chat_history": chat_history,
                    "context": rag_context,
                }
            )
            print(f"Bot: {response.content}")
            chat_history.append(("ai", response.content))

    with gr.Blocks() as demo:
        gr.Markdown(
            "# Harry PRT Chatbot\nAsk anything about Harry Potter and the Sorcerer's Stone!"
        )
        with gr.Accordion("Edit Persona, Directive, Output", open=False):
            persona_box = gr.Textbox(label="Persona", value=persona)
            directive_box = gr.Textbox(label="Directive", value=directive)
            output_box = gr.Textbox(label="Output", value=output)
        chatbot = gr.Chatbot(elem_id="main-chatbot")
        with gr.Row():
            msg = gr.Textbox(label="Your message", scale=4)
            send_btn = gr.Button("Send", scale=1)
        state = gr.State([])

        def respond(user_message, chat_history, persona_val, directive_val, output_val):
            bot_message, updated_history = chat_fn(
                user_message, chat_history, persona_val, directive_val, output_val
            )
            return "", updated_history

        send_btn.click(
            respond,
            [msg, chatbot, persona_box, directive_box, output_box],
            [msg, chatbot],
        )
        msg.submit(
            respond,
            [msg, chatbot, persona_box, directive_box, output_box],
            [msg, chatbot],
        )
        demo.launch(
            css="""
            #main-chatbot {
                height: 60vh !important;
                min-height: 30vh !important;
                max-height: 60vh !important;
                width: 100% !important;
            }
            .gradio-container {
                width: 100vw !important;
                min-height: 100vh !important;
                display: flex;
                flex-direction: column;
            }
            .gr-block.gr-chatbot {
                flex: 1 1 auto !important;
            }
            .gr-block.gr-row {
                flex-shrink: 0 !important;
            }
            textarea, input[type="text"] {
                max-width: 100% !important;
                box-sizing: border-box;
            }
            """
        )
    return


if __name__ == "__main__":
    main()
