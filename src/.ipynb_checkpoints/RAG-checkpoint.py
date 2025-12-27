import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

# class RAG_secondary_retrieval(nn.Module):
#     def __init__(self, input_channels=2, feature_dim=16, latent_dim=8, k=10, alpha=10.0):
#         super().__init__()
#         self.query_encoder = nn.Sequential(
#             nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
#             nn.BatchNorm3d(32),
#             nn.ReLU(),
#             nn.Conv3d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm3d(64),
#             nn.ReLU()
#         )
#         self.fc_query = nn.Linear(64, latent_dim)
#         self.k = k
#         self.alpha = alpha
#         self.index = faiss.IndexFlatL2(latent_dim)
#         self.label_store = []  # List[Tensor] storing labels (flip=1, not_flip=0)
#         self.embedding_store = []  # For optional backup

#     def context_voxel_mask(self, tumor_mask, window_size=5):
#         kernel = torch.ones((1, 1, window_size, window_size, window_size), device=tumor_mask.device)
#         context = F.conv3d(tumor_mask.float(), kernel, padding=window_size//2)
#         return (context > 0).float()

#     def encode_queries(self, bg_prob, ed_prob, context_mask):
#         x = torch.cat([bg_prob, ed_prob], dim=1)
#         feats = self.query_encoder(x).permute(0, 2, 3, 4, 1)  # (B, H, W, D, C)
#         selected = []
#         for b in range(feats.shape[0]):
#             selected_feats = feats[b][context_mask[b,0]==1]
#             selected.append(selected_feats)
#         selected = torch.cat(selected, dim=0)
#         latent = self.fc_query(selected)
#         return F.normalize(latent, dim=-1)  # (N_query, latent_dim)

#     def add_to_faiss(self, feature_embeddings, labels):
#         feature_embeddings = feature_embeddings.detach().cpu().numpy().astype('float32')
#         self.index.add(feature_embeddings)
#         self.label_store.append(labels.detach().cpu())
#         self.embedding_store.append(torch.tensor(feature_embeddings))

#     def retrieve_soft_knn(self, queries):
#         queries = queries.detach().cpu().numpy().astype('float32')
#         D, I = self.index.search(queries, self.k)
#         retrieved_probs = []
#         for dist, idxs in zip(D, I):
#             print("retreival in progress")
#             lbls = torch.stack([self.label_store[0][i] for i in idxs])  # assuming all labels in 1 tensor
#             dist = torch.tensor(dist)
#             weights = torch.exp(-self.alpha * dist)
#             prob = (weights * lbls.float()).sum() / (weights.sum() + 1e-8)
#             retrieved_probs.append(prob)
#         return torch.stack(retrieved_probs)

#     def forward(self, bg_prob, ed_prob, context_mask):
#         latent_queries = self.encode_queries(bg_prob, ed_prob, context_mask)
#         print("untill now network is working fine")
#         predictions = self.retrieve_soft_knn(latent_queries)
#         return predictions




# working code but with delay

# class RAG_secondary_retrieval(nn.Module):
#     def __init__(self, input_channels=2, feature_dim=16, latent_dim=8, k=10, alpha=10.0):
#         super().__init__()
#         self.query_encoder = nn.Sequential(
#             nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
#             nn.BatchNorm3d(32),
#             nn.ReLU(),
#             nn.Conv3d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm3d(64),
#             nn.ReLU()
#         )
#         self.fc_query = nn.Linear(64, latent_dim)
#         self.k = k
#         self.alpha = alpha

#         # Initialize FAISS index
#         self.index = faiss.IndexFlatL2(latent_dim)
        
#         # To store labels separately
#         self.label_store = []  # List of labels corresponding to features

#     def context_voxel_mask(self, tumor_mask, window_size=5):
#         kernel = torch.ones((1, 1, window_size, window_size, window_size), device=tumor_mask.device)
#         context = F.conv3d(tumor_mask.float(), kernel, padding=window_size // 2)
#         return (context > 0).float()

#     def encode_queries(self, bg_prob, ed_prob, context_mask):
#         """
#         Given bg_prob and ed_prob and context mask, encode features into low-dimension.
#         """
#         x = torch.cat([bg_prob, ed_prob], dim=1)
#         feats = self.query_encoder(x).permute(0, 2, 3, 4, 1)  # (B, H, W, D, C)

#         selected = []
#         for b in range(feats.shape[0]):
#             selected_feats = feats[b][context_mask[b, 0] == 1]
#             selected.append(selected_feats)
#         selected = torch.cat(selected, dim=0)  # (N, C)

#         latent = self.fc_query(selected)
#         return F.normalize(latent, dim=-1)  # (N_query, latent_dim)

#     def add_to_faiss(self, feature_embeddings, labels):
#         """
#         Add new features and corresponding labels into FAISS index.
#         """
#         feature_embeddings = feature_embeddings.detach().cpu().numpy().astype('float32')
#         labels = labels.detach().cpu()

#         self.index.add(feature_embeddings)
#         self.label_store.append(labels)

#     def retrieve_soft_knn(self, queries):
#         """
#         Retrieve k nearest neighbors using FAISS and do soft voting to produce predictions.
#         """
#         if self.index.ntotal == 0:
#             raise ValueError("FAISS index is empty! Cannot retrieve.")

#         queries = queries.detach().cpu().numpy().astype('float32')
#         D, I = self.index.search(queries, self.k)  # D: distances, I: indices

#         all_labels = torch.cat(self.label_store, dim=0)  # (Total_points,)
#         retrieved_probs = []

#         for dist, idxs in zip(D, I):
#             lbls = all_labels[idxs]  # Get labels of nearest neighbors
#             dist = torch.tensor(dist, dtype=torch.float32)
#             weights = torch.exp(-self.alpha * dist)  # Higher weight to closer neighbors

#             prob = (weights * lbls.float()).sum() / (weights.sum() + 1e-8)
#             retrieved_probs.append(prob)

#         return torch.stack(retrieved_probs)  # (N_queries,)

#     def forward(self, bg_prob, ed_prob, context_mask, add_mode=True, labels=None):
#         """
#         During forward pass:
#         - if add_mode == True: add to faiss
#         - if add_mode == False: retrieve from faiss
#         """
#         latent_queries = self.encode_queries(bg_prob, ed_prob, context_mask)

#         if add_mode:
#             if labels is None:
#                 raise ValueError("Labels must be provided when add_mode=True!")
#             self.add_to_faiss(latent_queries, labels)
#             return None  # No predictions during add phase
#         else:
#             predictions = self.retrieve_soft_knn(latent_queries)
#             return predictions





#Code with clustering

import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

class RAG_secondary_retrieval(nn.Module):
    def __init__(self, input_channels=2, latent_dim=8, k=10, alpha=10.0, index_type='ivfflat', nlist=500, nprobe=10, max_vectors=500):
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, latent_dim, kernel_size=1)
        )

        self.k = k
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.label_store = []
        self.max_vectors = max_vectors

        if index_type == 'flat':
            self.index = faiss.IndexFlatL2(latent_dim)
        elif index_type == 'lsh':
            self.index = faiss.IndexLSH(latent_dim, latent_dim * 8)
        elif index_type == 'ivfflat':
            quantizer = faiss.IndexFlatL2(latent_dim)
            self.index = faiss.IndexIVFFlat(quantizer, latent_dim, nlist)
            self.index.nprobe = nprobe
            self.requires_training = True
        else:
            raise ValueError("Unsupported index_type. Choose from 'flat', 'lsh', or 'ivfflat'.")

        self.gpu_index = self.index
        self.trained = not getattr(self, 'requires_training', False)

    def encode_queries(self, bg_prob, ed_prob, context_mask):
        x = torch.cat([bg_prob, ed_prob], dim=1)
        latent = self.query_encoder(x)
        latent = F.normalize(latent, dim=1)
        return latent

    def add_to_faiss(self, feature_embeddings, labels, context_mask):
        B, C, D, H, W = feature_embeddings.shape
        flat_feats = feature_embeddings.permute(0, 2, 3, 4, 1).reshape(-1, C).detach().cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.as_tensor() if hasattr(labels, "as_tensor") else labels
        if isinstance(context_mask, torch.Tensor):
            context_mask = context_mask.as_tensor() if hasattr(context_mask, "as_tensor") else context_mask

        flat_labels = labels.contiguous().view(-1).float().detach().cpu()
        flat_mask = context_mask.contiguous().view(-1).bool().detach().cpu().numpy()

        selected_feats = flat_feats[flat_mask]
        selected_labels = flat_labels[flat_mask]

        current_total = sum(len(l) for l in self.label_store)
        if current_total + selected_feats.shape[0] > self.max_vectors:
            print(f"[FAISS] Skipped adding {selected_feats.shape[0]} vectors. Limit {self.max_vectors} reached.")
            return

        if hasattr(self, 'requires_training') and self.requires_training and not self.trained:
            self.index.train(selected_feats)
            self.trained = True

        self.gpu_index.add(selected_feats)
        self.label_store.append(selected_labels)

        print(f"[FAISS] Added {selected_feats.shape[0]} vectors to index. Total stored: {current_total + selected_feats.shape[0]}")

    def retrieve_soft_knn(self, queries, batch_size=4096):
        B, C, D, H, W = queries.shape
        flat_queries = queries.permute(0, 2, 3, 4, 1).reshape(-1, C).detach().cpu().numpy()

        if not self.label_store:
            print("[FAISS] Retrieval skipped: index is empty.")
            return torch.zeros((B, 2, D, H, W), device='cuda')

        all_labels = torch.cat(self.label_store, dim=0).to(torch.float32)
        preds = []

        for i in range(0, flat_queries.shape[0], batch_size):
            q_batch = flat_queries[i:i+batch_size]
            Dists, Indices = self.gpu_index.search(q_batch, self.k)

            for dist, idxs in zip(Dists, Indices):
                lbls = all_labels[torch.tensor(idxs)]
                weights = torch.exp(-self.alpha * torch.tensor(dist, dtype=torch.float32))
                prob = (weights * lbls).sum() / (weights.sum() + 1e-8)
                preds.append(prob)

        out = torch.stack(preds).to('cuda')
        print("output shape is", out.shape)
        return out.view(B, 2, D, H, W)  # Removed channel dimension

    def forward(self, bg_prob, ed_prob, context_mask, add_mode=True, labels=None):
        latent = self.encode_queries(bg_prob, ed_prob, context_mask)
        if add_mode:
            self.add_to_faiss(latent, labels, context_mask)
            return None
        return self.retrieve_soft_knn(latent)