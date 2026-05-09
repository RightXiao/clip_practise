"""Gradio demo: text-to-image, image-to-image, zero-shot classification."""
import os
import gradio as gr
import torch
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
import open_clip


class ClipDemo:
    def __init__(self, checkpoint_path: str, image_dir: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=checkpoint_path
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        index_path = os.path.join(os.path.dirname(image_dir), "demo_index.faiss")
        meta_path = os.path.join(os.path.dirname(image_dir), "demo_meta.npy")
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            self.image_files = np.load(meta_path).tolist()
        else:
            self.index = self._build_index()
            faiss.write_index(self.index, index_path)
            np.save(meta_path, np.array(self.image_files))

    @torch.no_grad()
    def _extract_features(self, images):
        features = []
        for img_tensor in tqdm(images, desc="Extracting features"):
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                feat = torch.nn.functional.normalize(self.model.encode_image(img_tensor), dim=-1)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0).astype(np.float32)

    def _build_index(self):
        all_imgs = []
        for fname in self.image_files:
            img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
            all_imgs.append(self.preprocess(img))
        features = self._extract_features(all_imgs)
        index = faiss.IndexFlatIP(features.shape[1])
        index.add(features)
        return index

    @torch.no_grad()
    def text_to_image(self, query: str, top_k: int = 5):
        token_ids = self.tokenizer(query).to(self.device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            text_feat = torch.nn.functional.normalize(
                self.model.encode_text(token_ids), dim=-1
            )
        distances, indices = self.index.search(text_feat.cpu().numpy(), top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            results.append((img_path, f"Similarity: {dist:.3f}"))
        return results

    @torch.no_grad()
    def image_to_image(self, image: np.ndarray, top_k: int = 5):
        pil_img = Image.fromarray(image).convert("RGB")
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            img_feat = torch.nn.functional.normalize(
                self.model.encode_image(img_tensor), dim=-1
            )
        distances, indices = self.index.search(img_feat.cpu().numpy(), top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            results.append((img_path, f"Similarity: {dist:.3f}"))
        return results

    @torch.no_grad()
    def zeroshot_classify(self, image: np.ndarray, class_names_str: str) -> dict:
        class_names = [c.strip() for c in class_names_str.split(",") if c.strip()]
        if not class_names:
            return {"label": "Error", "scores": {"No classes provided": 1.0}}

        pil_img = Image.fromarray(image).convert("RGB")
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        prompts = [f"a photo of a {c}." for c in class_names]
        token_ids = torch.cat([self.tokenizer(p) for p in prompts]).to(self.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            img_feat = torch.nn.functional.normalize(
                self.model.encode_image(img_tensor), dim=-1
            )
            text_feat = torch.nn.functional.normalize(
                self.model.encode_text(token_ids), dim=-1
            )
            sim = (img_feat @ text_feat.t()).squeeze(0)
            probs = torch.softmax(sim * self.model.logit_scale.exp(), dim=0)

        scores = {name: probs[i].item() for i, name in enumerate(class_names)}
        return {"label": max(scores, key=scores.get), "scores": scores}


def create_demo(checkpoint_path: str, image_dir: str):
    demo_engine = ClipDemo(checkpoint_path, image_dir)

    with gr.Blocks(title="CLIP Demo") as app:
        gr.Markdown("# CLIP 图文检索 & 零样本分类 Demo")

        with gr.Tab("文字搜图"):
            text_input = gr.Textbox(label="输入描述文字", placeholder="a cat sitting on a chair")
            top_k_slider = gr.Slider(1, 20, value=5, step=1, label="返回数量")
            text_btn = gr.Button("搜索")
            text_gallery = gr.Gallery(label="检索结果")
            text_btn.click(demo_engine.text_to_image, inputs=[text_input, top_k_slider], outputs=text_gallery)

        with gr.Tab("以图搜图"):
            image_input = gr.Image(label="上传图片", type="numpy")
            img_k_slider = gr.Slider(1, 20, value=5, step=1, label="返回数量")
            img_btn = gr.Button("搜索")
            img_gallery = gr.Gallery(label="相似图片")
            img_btn.click(demo_engine.image_to_image, inputs=[image_input, img_k_slider], outputs=img_gallery)

        with gr.Tab("零样本分类"):
            classify_img = gr.Image(label="上传图片", type="numpy")
            class_input = gr.Textbox(label="类别 (逗号分隔)", value="cat, dog, bird, car, tree")
            classify_btn = gr.Button("分类")
            classify_output = gr.Label(label="分类结果")
            classify_btn.click(demo_engine.zeroshot_classify, inputs=[classify_img, class_input], outputs=classify_output)

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app = create_demo(args.checkpoint, args.image_dir)
    app.launch(server_name="0.0.0.0", server_port=args.port)
