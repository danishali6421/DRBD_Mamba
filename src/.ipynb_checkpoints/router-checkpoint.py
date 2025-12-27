import os
import faiss
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
from sklearn.cluster import KMeans
from torch import nn


class FAISSRouterBuilder:
    def __init__(self, max_per_class=10):
        self.max_per_class = max_per_class
        self.classes = [0, 1, 2]  # 0 = all labels, 1 = TC missing, 2 = ET missing

    def compute_router_class(self, gt_mask):
        labels = gt_mask
        # print("labels", labels)
        if labels == 0:
            return 0
        elif labels == 1:
            return 1
        elif labels == 2:
            return 2
        # else:
        #     return 0  # fallback if ambiguous

    def should_build_index(self, mask_list, x4_list, device):
        """
        Returns True if all classes (0, 1, 2) have at least `max_per_class` samples
        """
        count_by_class = {k: 0 for k in self.classes}
        with torch.no_grad():
            for x4, mask in zip(x4_list, mask_list):
                x4 = x4.to(device)
                mask = mask.to(device)
                for i in range(x4.shape[0]):
                    cls = self.compute_router_class(mask[i])
                    count_by_class[cls] += 1
                    # print("count_by_class", count_by_class)
        return all(count_by_class[c] >= self.max_per_class for c in self.classes)

    
    def extract_descriptor(self, x4):
        x = rearrange(x4, 'b c h w d -> b (h w d) c')
        return x.mean(dim=1)  # global average pooling

    

    def build_index_from_raw(self, x4_list, mask_list, device, save_index_path="router_faiss_index.idx", save_label_path="router_labels.npy"):
        assert len(x4_list) == len(mask_list), "Mismatch in number of features and masks"
        descriptors_by_class = {k: [] for k in self.classes}
    
        with torch.no_grad():
            for x4, mask in zip(x4_list, mask_list):
                x4 = x4.to(device)
                mask = mask.to(device)
                desc = x4  # [B, C, H, W, D] or [B, C] depending on quantized
    
                for i in range(desc.shape[0]):
                    cls = self.compute_router_class(mask[i])
                    descriptors_by_class[cls].append(desc[i].cpu().numpy())
    
        # Now process each class individually
        all_desc = []
        all_labels = []
    
        for cls in self.classes:
            print("cls is", cls)
            samples = descriptors_by_class[cls]
            count = len(samples)
            print("count is", count)
    
            if count == 0:
                print(f"⚠️ Class {cls} has 0 samples. Skipping.")
                continue
            elif count > self.max_per_class:
                print(f"⚠️ Class {cls} has {count} samples. Reducing to {self.max_per_class} using KMeans.")
                kmeans = KMeans(n_clusters=self.max_per_class, random_state=42).fit(np.stack(samples))
                selected = kmeans.cluster_centers_
            elif count < self.max_per_class:
                print(f"⚠️ Class {cls} has only {count} samples. Padding to {self.max_per_class}.")
                avg = np.mean(samples, axis=0)
                selected = samples + [avg] * (self.max_per_class - count)
            else:
                selected = samples
    
            all_desc.extend(selected)
            all_labels.extend([cls] * self.max_per_class)
    
        all_desc = np.stack(all_desc).astype(np.float32)
        all_labels = np.array(all_labels).astype(np.int64)
    
        index = faiss.IndexFlatL2(all_desc.shape[1])
        index.add(all_desc)
        faiss.write_index(index, save_index_path)
        np.save(save_label_path, all_labels)
    
        print(f"✅ FAISS index saved to {save_index_path}, labels saved to {save_label_path}")
    
    

class FAISSRouterRetriever(torch.nn.Module):
    def __init__(self, index_path="router_faiss_index.idx", label_path="router_labels.npy", top_k=5):
        super().__init__()
        self.index_path = index_path
        self.label_path = label_path
        self.top_k = top_k
        self.faiss_index = None
        self.router_labels = None

    def extract_descriptor(self, x4):
        x = rearrange(x4, 'b c h w d -> b (h w d) c')
        return x.mean(dim=1)

    def load_index(self):
        if self.faiss_index is None:
            self.faiss_index = faiss.read_index(self.index_path)
            self.router_labels = np.load(self.label_path)

            unique, counts = np.unique(self.router_labels, return_counts=True)
            print("🧠 Label distribution in FAISS index:")
            for u, c in zip(unique, counts):
                print(f"Class {u}: {c} samples")
            print(f"\n📐 FAISS index contains {self.faiss_index.ntotal} descriptors")
            print(f"Each descriptor has dimension: {self.faiss_index.d}")
        
    def forward(self, x4):
        """
        Returns soft logits (vote counts per class) for each input sample.
        """
        self.load_index()
        with torch.no_grad():
            descriptors = x4.cpu().numpy()
            _, I = self.faiss_index.search(descriptors, self.top_k)
            logits = []
            for idxs in I:
                votes = self.router_labels[idxs]
                counts = np.bincount(votes, minlength=3)  # Ensure length = num_classes
                logits.append(counts)
            logits = torch.tensor(logits, device=x4.device, dtype=torch.float32)
            return logits  # shape: [B, num_classes]

    def forward_with_neighbors(self, x4):
        """
        Returns both soft logits and neighbor indices.
        """
        self.load_index()
        with torch.no_grad():
            descriptors = x4.cpu().numpy()
            _, I = self.faiss_index.search(descriptors, self.top_k)
            logits = []
            for idxs in I:
                votes = self.router_labels[idxs]
                counts = np.bincount(votes, minlength=3)
                logits.append(counts)
            logits = torch.tensor(logits, device=x4.device, dtype=torch.float32)
            return logits, I




# class FAISSRouterBuilder:
#     def __init__(self, max_per_class=10):
#         self.max_per_class = max_per_class
#         self.classes = [0, 1, 2]  # 0 = all labels, 1 = TC missing, 2 = ET missing

#     def compute_router_class(self, gt_mask):
#         labels = gt_mask
#         if labels == 0:
#             return 0
#         elif labels == 1:
#             return 1
#         elif labels == 2:
#             return 2

#     def should_build_index(self, mask_list, x4_list, device):
#         count_by_class = {k: 0 for k in self.classes}
#         with torch.no_grad():
#             for x4, mask in zip(x4_list, mask_list):
#                 x4 = x4.to(device)
#                 mask = mask.to(device)
#                 for i in range(x4.shape[0]):
#                     cls = self.compute_router_class(mask[i])
#                     count_by_class[cls] += 1
#         return all(count_by_class[c] >= self.max_per_class for c in self.classes)

#     def extract_descriptor(self, x4):
#         x = rearrange(x4, 'b c h w d -> b (h w d) c')
#         return x.mean(dim=1)  # global average pooling

#     def build_index_from_raw(self, x4_list, mask_list, device, save_index_path="router_faiss_index.idx", save_label_path="router_labels.npy", save_vector_path="router_vectors.npy"):
#         assert len(x4_list) == len(mask_list), "Mismatch in number of features and masks"
#         descriptors_by_class = {k: [] for k in self.classes}

#         with torch.no_grad():
#             for x4, mask in zip(x4_list, mask_list):
#                 x4 = x4.to(device)
#                 mask = mask.to(device)
#                 desc = x4

#                 for i in range(desc.shape[0]):
#                     cls = self.compute_router_class(mask[i])
#                     descriptors_by_class[cls].append(desc[i].cpu().numpy())

#         all_desc = []
#         all_labels = []

#         for cls in self.classes:
#             print("cls is", cls)
#             samples = descriptors_by_class[cls]
#             count = len(samples)
#             print("count is", count)

#             if count == 0:
#                 print(f"⚠️ Class {cls} has 0 samples. Skipping.")
#                 continue
#             elif count > self.max_per_class:
#                 print(f"⚠️ Class {cls} has {count} samples. Reducing to {self.max_per_class} using KMeans.")
#                 kmeans = KMeans(n_clusters=self.max_per_class, random_state=42).fit(np.stack(samples))
#                 selected = kmeans.cluster_centers_
#             elif count < self.max_per_class:
#                 print(f"⚠️ Class {cls} has only {count} samples. Padding to {self.max_per_class}.")
#                 avg = np.mean(samples, axis=0)
#                 selected = samples + [avg] * (self.max_per_class - count)
#             else:
#                 selected = samples

#             all_desc.extend(selected)
#             all_labels.extend([cls] * self.max_per_class)

#         all_desc = np.stack(all_desc).astype(np.float32)
#         all_labels = np.array(all_labels).astype(np.int64)

#         index = faiss.IndexFlatL2(all_desc.shape[1])
#         index.add(all_desc)
#         faiss.write_index(index, save_index_path)
#         np.save(save_label_path, all_labels)
#         np.save(save_vector_path, all_desc)  # Added saving of vectors for retrieval

#         print(f"✅ FAISS index saved to {save_index_path}, labels saved to {save_label_path}, vectors saved to {save_vector_path}")





# class FAISSRouterRetriever(nn.Module):
#     def __init__(self, index_path="router_faiss_index.idx", label_path="router_labels.npy", vector_path="router_vectors.npy", top_k=5):
#         super().__init__()
#         self.index_path = index_path
#         self.label_path = label_path
#         self.vector_path = vector_path
#         self.top_k = top_k
#         self.faiss_index = None
#         self.router_labels = None
#         self.vector_store = None

#     def extract_descriptor(self, x4):
#         x = rearrange(x4, 'b c h w d -> b (h w d) c')
#         return x.mean(dim=1)  # [B, D]

#     def load_index(self):
#         if self.faiss_index is None:
#             self.faiss_index = faiss.read_index(self.index_path)
#             self.router_labels = np.load(self.label_path)
#             self.vector_store = np.load(self.vector_path)

#             unique, counts = np.unique(self.router_labels, return_counts=True)
#             print("🧠 Label distribution in FAISS index:")
#             for u, c in zip(unique, counts):
#                 print(f"Class {u}: {c} samples")
#             print(f"\n📐 FAISS index contains {self.faiss_index.ntotal} descriptors")
#             print(f"Each descriptor has dimension: {self.faiss_index.d}")

#     def forward(self, x4):
#         """
#         Returns soft logits (vote counts per class) for each input sample.
#         """
#         self.load_index()
#         with torch.no_grad():
#             descriptors = x4.cpu().numpy()
#             _, I = self.faiss_index.search(descriptors, self.top_k)
#             logits = []
#             for idxs in I:
#                 votes = self.router_labels[idxs]
#                 counts = np.bincount(votes, minlength=3)
#                 logits.append(counts)
#             logits = torch.tensor(logits, device=x4.device, dtype=torch.float32)
#             return logits

#     def retrieve_topk_embeddings(self, x4):
#         """
#         Returns top_k embedding vectors [B, top_k, D] for each input query x4.
#         """
#         self.load_index()
#         with torch.no_grad():
#             descriptors = x4.cpu().numpy()  # [B, D]
#             _, I = self.faiss_index.search(descriptors, self.top_k)
#             topk_embeddings = self.vector_store[I]  # [B, top_k, D]
#             print("topk_embeddings", topk_embeddings.shape)
#             return torch.tensor(topk_embeddings, dtype=torch.float32, device=x4.device)
