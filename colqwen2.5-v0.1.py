import streamlit as st
import torch
import numpy as np
import uuid
import io
import fitz  # PyMuPDF
from PIL import Image
from qdrant_client import QdrantClient, models
from colpali_engine.models import ColQwen2, ColQwen2Processor

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")
collection_name = "pdf_collection"

# Check if the collection already exists; create only if it doesn't.
try:
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "original": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),  # HNSW turned off
                ),
                "mean_pooling_columns": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
                "mean_pooling_rows": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            },
        )
    else:
        st.write("Collection already exists; skipping creation.")
except Exception as e:
    st.write("Error during collection creation:", e)

# Load ColQwen model and processor (cached for faster reloads)
@st.cache_resource
def load_model():
    colqwen_model = ColQwen2.from_pretrained(
        "vidore/colqwen2.5-v0.2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # or "mps" if on Apple Silicon
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()
    colqwen_processor = ColQwen2Processor.rom_pretrained("vidore/colqwen2.5-v0.1")
    return colqwen_model, colqwen_processor


colqwen_model, colqwen_processor = load_model()


# Alternative PDF conversion using PyMuPDF (avoids needing Poppler)
def convert_pdf_to_images(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        pix = page.get_pixmap()
        # Convert pixmap to PNG image bytes, then open with PIL
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images


# Helper functions remain unchanged
def get_patches(image_size, model_processor, model):
    # For ColQwen, we use its patch size and spatial merge size
    return model_processor.get_n_patches(
        image_size,
        patch_size=model.patch_size,
        spatial_merge_size=model.spatial_merge_size,
    )


def embed_and_mean_pool_batch(image_batch, model_processor, model):
    # Embed images using ColQwen
    with torch.no_grad():
        processed_images = model_processor.process_images(image_batch).to(model.device)
        image_embeddings = model(**processed_images)
    image_embeddings_batch = image_embeddings.cpu().float().numpy().tolist()

    pooled_by_rows_batch = []
    pooled_by_columns_batch = []
    for image_embedding, tokenized_image, image in zip(
        image_embeddings, processed_images.input_ids, image_batch
    ):
        x_patches, y_patches = get_patches(image.size, model_processor, model)
        image_tokens_mask = tokenized_image == model_processor.image_token_id
        image_tokens = image_embedding[image_tokens_mask].view(
            x_patches, y_patches, model.dim
        )
        pooled_by_rows = torch.mean(image_tokens, dim=0)
        pooled_by_columns = torch.mean(image_tokens, dim=1)
        image_token_idxs = torch.nonzero(image_tokens_mask.int(), as_tuple=False)
        first_image_token_idx = image_token_idxs[0].cpu().item()
        last_image_token_idx = image_token_idxs[-1].cpu().item()
        prefix_tokens = image_embedding[:first_image_token_idx]
        postfix_tokens = image_embedding[last_image_token_idx + 1 :]
        pooled_by_rows = torch.cat(
            (prefix_tokens, pooled_by_rows, postfix_tokens), dim=0
        )
        pooled_by_columns = torch.cat(
            (prefix_tokens, pooled_by_columns, postfix_tokens), dim=0
        )
        pooled_by_rows_batch.append(pooled_by_rows.cpu().float().numpy().tolist())
        pooled_by_columns_batch.append(pooled_by_columns.cpu().float().numpy().tolist())
    return image_embeddings_batch, pooled_by_rows_batch, pooled_by_columns_batch


def upload_batch(
    original_batch,
    pooled_by_rows_batch,
    pooled_by_columns_batch,
    payload_batch,
    collection_name,
):
    try:
        client.upload_collection(
            collection_name=collection_name,
            vectors={
                "mean_pooling_columns": pooled_by_columns_batch,
                "original": original_batch,
                "mean_pooling_rows": pooled_by_rows_batch,
            },
            payload=payload_batch,
            ids=[str(uuid.uuid4()) for _ in range(len(original_batch))],
        )
    except Exception as e:
        st.write(f"Error during upsert: {e}")


def batch_embed_query(query_batch, model_processor, model):
    with torch.no_grad():
        processed_queries = model_processor.process_queries(query_batch).to(
            model.device
        )
        query_embeddings_batch = model(**processed_queries)
    return query_embeddings_batch.cpu().float().numpy()


def reranking_search_batch(
    query_batch, collection_name, search_limit=20, prefetch_limit=200
):
    search_queries = [
        models.QueryRequest(
            query=query,
            prefetch=[
                models.Prefetch(
                    query=query, limit=prefetch_limit, using="mean_pooling_columns"
                ),
                models.Prefetch(
                    query=query, limit=prefetch_limit, using="mean_pooling_rows"
                ),
            ],
            limit=search_limit,
            with_payload=True,
            with_vector=False,
            using="original",
        )
        for query in query_batch
    ]
    return client.query_batch_points(
        collection_name=collection_name, requests=search_queries
    )


# ---------------------- Streamlit App UI ---------------------- #
st.title("PDF Upload and Query Application (ColQwen Experiment)")

# --- PDF Upload Section --- #
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    try:
        # Use PyMuPDF instead of pdf2image to convert PDF pages to images
        pages = convert_pdf_to_images(file_bytes)
        st.write(f"Converted PDF to {len(pages)} image pages.")
    except Exception as e:
        st.error(f"Error converting PDF: {e}")
        pages = []

    if pages:
        # Display thumbnails of each page
        st.image(pages, width=150, caption=[f"Page {i+1}" for i in range(len(pages))])
        if st.button("Process and Upload PDF"):
            original_embeddings = []
            pooled_rows = []
            pooled_columns = []
            payloads = []
            progress_bar = st.progress(0)
            for i, page in enumerate(pages):
                image = page.convert("RGB")
                image_batch = [image]
                try:
                    orig_batch, pooled_rows_batch, pooled_columns_batch = (
                        embed_and_mean_pool_batch(
                            image_batch, colqwen_processor, colqwen_model
                        )
                    )
                    original_embeddings.extend(orig_batch)
                    pooled_rows.extend(pooled_rows_batch)
                    pooled_columns.extend(pooled_columns_batch)
                    payloads.append({"source": "uploaded_pdf", "index": i})
                except Exception as e:
                    st.write(f"Error processing page {i+1}: {e}")
                    continue
                progress_bar.progress((i + 1) / len(pages))
            upload_batch(
                np.asarray(original_embeddings, dtype=np.float32),
                np.asarray(pooled_rows, dtype=np.float32),
                np.asarray(pooled_columns, dtype=np.float32),
                payloads,
                collection_name,
            )
            st.success("PDF processing and upload complete!")
            st.session_state["pdf_pages"] = pages

# --- Query Section --- #
if "pdf_pages" in st.session_state:
    st.header("Ask a Question")
    query_input = st.text_input("Enter your query:")
    if st.button("Search") and query_input:
        query_embedding = batch_embed_query(
            [query_input], colqwen_processor, colqwen_model
        )
        results = reranking_search_batch(query_embedding, collection_name)
        if results and results[0].points:
            best_match = results[0].points[0]
            page_index = best_match.payload.get("index", None)
            st.write(f"Best matching page index: {page_index}")
            if page_index is not None and page_index < len(
                st.session_state["pdf_pages"]
            ):
                st.image(
                    st.session_state["pdf_pages"][page_index],
                    caption=f"Page {page_index + 1}",
                )
        else:
            st.write("No matching pages found.")
