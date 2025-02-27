import streamlit as st
import io
import uuid
import base64
import numpy as np
from PIL import Image
import torch
import fitz  # PyMuPDF
from qdrant_client import QdrantClient, models
from colpali_engine.models import ColPali, ColPaliProcessor


# ---------------------------
# Helper: Convert PIL Image to base64
# ---------------------------
def image_to_base64(image, format="PNG"):
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


# ---------------------------
# Setup: Qdrant and ColPali Model
# ---------------------------
client = QdrantClient(url="http://localhost:6333")

colpali_model = ColPali.from_pretrained(
    "vidore/colpali-v1.3",
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # change to "cuda:0" or "mps" if GPU is available
).eval()

colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")

# Temporary collection name for the uploaded PDF
COLLECTION_NAME = "colpali_pdf_upload"


# ---------------------------
# Create Collection if It Doesn't Exist
# ---------------------------
def create_collection():
    try:
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
    except Exception:
        collection_names = []

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "original": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=0
                    ),  # disable HNSW for original vectors
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


# Initialize or reuse the collection
create_collection()


# ---------------------------
# Embedding & Retrieval Helper Functions
# (Functions remain unchanged)
# ---------------------------
def get_patches(image_size, model_processor, model, model_name="colPali"):
    return model_processor.get_n_patches(image_size, patch_size=model.patch_size)


def embed_and_mean_pool_batch(
    image_batch, model_processor, model, model_name="colPali"
):
    with torch.no_grad():
        processed_images = model_processor.process_images(image_batch).to(model.device)
        image_embeddings = model(**processed_images)
    image_embeddings_batch = image_embeddings.cpu().float().numpy().tolist()
    pooled_by_rows_batch = []
    pooled_by_columns_batch = []
    for image_embedding, tokenized_image, image in zip(
        image_embeddings, processed_images.input_ids, image_batch
    ):
        x_patches, y_patches = get_patches(
            image.size, model_processor, model, model_name
        )
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
        pooled_by_rows = (
            torch.cat((prefix_tokens, pooled_by_rows, postfix_tokens), dim=0)
            .cpu()
            .float()
            .numpy()
            .tolist()
        )
        pooled_by_columns = (
            torch.cat((prefix_tokens, pooled_by_columns, postfix_tokens), dim=0)
            .cpu()
            .float()
            .numpy()
            .tolist()
        )
        pooled_by_rows_batch.append(pooled_by_rows)
        pooled_by_columns_batch.append(pooled_by_columns)
    return image_embeddings_batch, pooled_by_rows_batch, pooled_by_columns_batch


def upload_batch(
    original_batch,
    pooled_by_rows_batch,
    pooled_by_columns_batch,
    payload_batch,
    collection_name,
):
    client.upload_collection(
        collection_name=collection_name,
        vectors={
            "original": original_batch,
            "mean_pooling_rows": pooled_by_rows_batch,
            "mean_pooling_columns": pooled_by_columns_batch,
        },
        payload=payload_batch,
        ids=[str(uuid.uuid4()) for _ in range(len(original_batch))],
    )


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


# ---------------------------
# PDF Processing: Convert PDF to PIL Images
# ---------------------------
def pdf_to_images(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    return images


# ---------------------------
# Session State Initialization
# ---------------------------
if "pdf_indexed" not in st.session_state:
    st.session_state.pdf_indexed = False
if "pages" not in st.session_state:
    st.session_state.pages = None

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.title("PDF Retrieval with ColPali and Qdrant")
st.write(
    "Upload a PDF to index its pages. You can query your vector database anytime—even if you don't update the file."
)

# File upload and (re)indexing section
uploaded_pdf = st.file_uploader("Choose a PDF file to (re)index", type=["pdf"])

if uploaded_pdf is not None:
    st.info("Processing PDF...")
    pdf_bytes = uploaded_pdf.getvalue()
    pdf_file = io.BytesIO(pdf_bytes)
    pages = pdf_to_images(pdf_file)
    st.session_state.pages = pages
    st.success(f"PDF processed: {len(pages)} pages detected.")
    st.info("Displaying all pages...")
    thumbnails = []
    for page in pages:
        thumb = page.copy()
        thumb.thumbnail((200, 2000))  # Resize width to 200px
        thumbnails.append(thumb)
    thumbnail_html = '<div style="overflow-x: auto; white-space: nowrap;">'
    for idx, thumb in enumerate(thumbnails):
        img_b64 = image_to_base64(thumb)
        thumbnail_html += (
            f'<div style="display: inline-block; margin-right: 10px; text-align: center;">'
            f'<img src="data:image/png;base64,{img_b64}" style="width: 200px; cursor: pointer;" '
            f"onclick=\"window.parent.postMessage('enlarge:{idx}', '*')\"><br>Page {idx + 1}</div>"
        )
    thumbnail_html += "</div>"
    st.markdown(thumbnail_html, unsafe_allow_html=True)
    st.info("Indexing PDF pages...")
    progress_bar = st.progress(0)
    for idx, page in enumerate(pages):
        original_batch, pooled_by_rows_batch, pooled_by_columns_batch = (
            embed_and_mean_pool_batch(
                [page], colpali_processor, colpali_model, model_name="colPali"
            )
        )
        payload = [{"page_number": idx}]
        upload_batch(
            np.asarray(original_batch, dtype=np.float32),
            np.asarray(pooled_by_rows_batch, dtype=np.float32),
            np.asarray(pooled_by_columns_batch, dtype=np.float32),
            payload,
            COLLECTION_NAME,
        )
        progress_bar.progress((idx + 1) / len(pages))
    st.success("Indexing complete!")
    st.session_state.pdf_indexed = True

# Display thumbnails and allow enlargement if pages are available
if st.session_state.pages is not None:
    pages = st.session_state.pages
    st.markdown("<h3>Page Thumbnails</h3>", unsafe_allow_html=True)
    thumbnails = []
    for page in pages:
        thumb = page.copy()
        thumb.thumbnail((200, 2000))  # Resize width to 200px
        thumbnails.append(thumb)
    thumbnail_html = '<div style="overflow-x: auto; white-space: nowrap;">'
    for idx, thumb in enumerate(thumbnails):
        img_b64 = image_to_base64(thumb)
        thumbnail_html += (
            f'<div style="display: inline-block; margin-right: 10px; text-align: center;">'
            f'<img src="data:image/png;base64,{img_b64}" style="width: 200px; cursor: pointer;" '
            f"onclick=\"window.parent.postMessage('enlarge:{idx}', '*')\"><br>Page {idx + 1}</div>"
        )
    thumbnail_html += "</div>"
    st.markdown(thumbnail_html, unsafe_allow_html=True)
    selected_page = st.selectbox(
        "Select a page to enlarge", list(range(1, len(pages) + 1))
    )
    st.image(
        pages[selected_page - 1],
        caption=f"Enlarged View - Page {selected_page}",
        use_column_width=True,
    )

# ---------------------------
# Query Interface (Always Available)
# ---------------------------
st.markdown("### Query the PDF Data")
question = st.text_input("Enter a question about the PDF:")

if question:
    st.info("Retrieving the most relevant page from the vector database...")
    query_embedding = batch_embed_query([question], colpali_processor, colpali_model)
    results = reranking_search_batch(query_embedding, COLLECTION_NAME)
    if results and results[0].points:
        top_result = results[0].points[0]
        page_idx = top_result.payload.get("page_number", None)
        st.write(
            f"Top matching page: {page_idx + 1 if page_idx is not None else 'N/A'}"
        )
        # Display the page image if available; otherwise just show the page number.
        if (
            st.session_state.pages
            and page_idx is not None
            and page_idx < len(st.session_state.pages)
        ):
            st.image(
                st.session_state.pages[page_idx],
                caption=f"Page {page_idx + 1}",
                use_column_width=True,
            )
        else:
            st.warning(
                "Page image not available (it may not be loaded in this session)."
            )
    else:
        st.warning("No relevant pages found.")
