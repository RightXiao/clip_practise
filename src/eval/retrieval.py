"""Image-Text retrieval evaluation with Recall@K."""
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def evaluate_retrieval(model, loader, device: torch.device) -> dict:
    """Compute image->text and text->image Recall@K.

    Returns dict with: i2t_r1, i2t_r5, i2t_r10, t2i_r1, t2i_r5, t2i_r10
    """
    model.eval()
    all_image_embeds = []
    all_text_embeds = []

    for images, token_ids in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        token_ids = token_ids.to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(token_ids)

        all_image_embeds.append(F.normalize(img_emb, dim=-1).cpu())
        all_text_embeds.append(F.normalize(txt_emb, dim=-1).cpu())

    image_embeds = torch.cat(all_image_embeds, dim=0)  # (N, D)
    text_embeds = torch.cat(all_text_embeds, dim=0)    # (N, D)

    sim = image_embeds @ text_embeds.t()  # (N, N)
    i2t = _recall_at_k(sim, ks=[1, 5, 10])
    t2i = _recall_at_k(sim.t(), ks=[1, 5, 10])

    return {
        "i2t_r1": i2t[0], "i2t_r5": i2t[1], "i2t_r10": i2t[2],
        "t2i_r1": t2i[0], "t2i_r5": t2i[1], "t2i_r10": t2i[2],
    }


def _recall_at_k(sim_matrix: torch.Tensor, ks: list[int]) -> list[float]:
    """sim_matrix: (num_queries, num_items). True match is on diagonal."""
    n = sim_matrix.size(0)
    labels = torch.arange(n)
    _, indices = sim_matrix.topk(max(ks), dim=1)
    correct = indices == labels.unsqueeze(1)
    return [correct[:, :k].any(dim=1).float().mean().item() for k in ks]
