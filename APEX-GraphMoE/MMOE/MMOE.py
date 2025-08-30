# multi_mmoe_heterogeneous_topk.py - Top-K Anti-Polarization Diversified Heterogeneous Expert Multimodal MMoE Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
import json
import math


class TopKGate(nn.Module):
    """Top-K Gating Network - Dynamic selection mechanism to prevent expert polarization"""

    #APG coding will be published after accept

class CNNImageExpert(nn.Module):
    """CNN Image Expert - Convolutional structure for image feature processing"""

    def __init__(self, image_dim=256, expert_hidden_dim=512, dropout=0.2):
        super().__init__()
        self.expert_type = "cnn_image"

        self.feature_map_size = int(math.sqrt(image_dim))
        if self.feature_map_size * self.feature_map_size != image_dim:
            self.input_projection = nn.Linear(image_dim, 16 * 16)
            self.feature_map_size = 16
        else:
            self.input_projection = None

        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.feature_projection = nn.Sequential(
            nn.Linear(512, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_hidden_dim)
        )

        if self.input_projection:
            self.residual_projection = nn.Linear(16 * 16, expert_hidden_dim)
        else:
            self.residual_projection = nn.Linear(image_dim, expert_hidden_dim)

    def forward(self, image_emb):
        batch_size = image_emb.size(0)

        if self.input_projection:
            x = self.input_projection(image_emb)
            residual_input = x
        else:
            x = image_emb
            residual_input = image_emb

        x = x.view(batch_size, 1, self.feature_map_size, self.feature_map_size)

        cnn_features = self.cnn_backbone(x)
        pooled_features = self.global_pool(cnn_features)
        pooled_features = pooled_features.view(batch_size, -1)

        main_output = self.feature_projection(pooled_features)
        residual = self.residual_projection(residual_input)

        output = main_output + residual * 0.1

        return output


class TransformerTextExpert(nn.Module):
    """Transformer Text Expert - Standard Transformer for text feature processing"""

    def __init__(self, text_dim=3840, expert_hidden_dim=512, dropout=0.2):
        super().__init__()
        self.expert_type = "transformer_text"
        self.text_fields = 5
        self.field_dim = 768

        self.position_embedding = nn.Parameter(torch.randn(self.text_fields, self.field_dim))
        self.input_projection = nn.Linear(self.field_dim, expert_hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=expert_hidden_dim,
            nhead=8,
            dim_feedforward=expert_hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=expert_hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.query_vector = nn.Parameter(torch.randn(1, expert_hidden_dim))

        self.output_projection = nn.Sequential(
            nn.LayerNorm(expert_hidden_dim),
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_hidden_dim)
        )

        self.residual_projection = nn.Linear(text_dim, expert_hidden_dim)

    def forward(self, text_emb):
        batch_size = text_emb.size(0)

        text_fields = text_emb.view(batch_size, self.text_fields, self.field_dim)
        text_fields = text_fields + self.position_embedding.unsqueeze(0)

        projected_inputs = self.input_projection(text_fields)
        transformer_output = self.transformer_encoder(projected_inputs)

        query = self.query_vector.expand(batch_size, -1, -1)
        pooled_output, attention_weights = self.attention_pooling(
            query, transformer_output, transformer_output
        )

        pooled_output = pooled_output.squeeze(1)
        main_output = self.output_projection(pooled_output)
        residual = self.residual_projection(text_emb)

        output = main_output + residual * 0.1

        return output


class IDExpert(nn.Module):
    """Specialized expert for ID feature processing with factorization"""

    def __init__(self, id_dim=928, expert_hidden_dim=512, dropout=0.2):
        super().__init__()
        self.expert_type = "id"
        self.num_fields = 28
        self.field_dim = 32

        self.field_importance = nn.Parameter(torch.ones(self.num_fields))

        self.user_item_processor = nn.Sequential(
            nn.Linear(self.field_dim * 2, expert_hidden_dim // 2),
            nn.BatchNorm1d(expert_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.deep_features_processor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.field_dim, expert_hidden_dim // 8),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(26)
        ])

        self.interaction_layer = nn.Sequential(
            nn.Linear(expert_hidden_dim // 2 + expert_hidden_dim // 8 * 26, expert_hidden_dim),
            nn.BatchNorm1d(expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.deep_network = nn.Sequential(
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.BatchNorm1d(expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.BatchNorm1d(expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.output_projection = nn.Linear(expert_hidden_dim, expert_hidden_dim)

    def forward(self, id_emb):
        batch_size = id_emb.size(0)

        id_fields = id_emb.view(batch_size, self.num_fields, self.field_dim)

        user_item_concat = torch.cat([id_fields[:, 0, :], id_fields[:, 1, :]], dim=1)
        user_item_out = self.user_item_processor(user_item_concat)

        deep_feature_outputs = []
        for i, processor in enumerate(self.deep_features_processor):
            field_idx = i + 2
            weighted_field = id_fields[:, field_idx, :] * self.field_importance[field_idx]
            deep_out = processor(weighted_field)
            deep_feature_outputs.append(deep_out)

        deep_features_concat = torch.cat(deep_feature_outputs, dim=1)
        combined_features = torch.cat([user_item_out, deep_features_concat], dim=1)

        interaction_out = self.interaction_layer(combined_features)
        deep_out = self.deep_network(interaction_out)

        output = self.output_projection(deep_out) + interaction_out * 0.1

        return output


class LightGCNGraphExpert(nn.Module):
    """Optimized LightGCN Graph Expert with batch precomputation and caching"""

    def __init__(self, expert_hidden_dim=512, dropout=0.2, gcn_logs_dir="gcn_logs_incremental"):
        super().__init__()
        self.expert_type = "graph"
        self.expert_hidden_dim = expert_hidden_dim
        self.gcn_logs_dir = gcn_logs_dir

        self.scene_user_embeddings_gpu = {}
        self.scene_item_embeddings_gpu = {}
        self.scene_mappings = {}
        self.scene_avg_user_emb = {}
        self.scene_avg_item_emb = {}

        self.device = None
        self.embeddings_loaded_on_gpu = False
        self.graph_emb_dim = 0
        self.lightgcn_enabled = True

        self._load_all_scene_embeddings()

        if self.graph_emb_dim > 0 and self.lightgcn_enabled:
            self.graph_projection = nn.Sequential(
                nn.Linear(self.graph_emb_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, expert_hidden_dim)
            )
        else:
            self.graph_projection = nn.Sequential(
                nn.Linear(expert_hidden_dim // 4, expert_hidden_dim),
                nn.ReLU()
            )
            self.lightgcn_enabled = False

    def _load_all_scene_embeddings(self):
        """Load all graph embeddings to GPU with precomputation"""
        if not os.path.exists(self.gcn_logs_dir):
            self.lightgcn_enabled = False
            return

        scene_dirs = [d for d in os.listdir(self.gcn_logs_dir) if d.startswith('scene_')]

        if not scene_dirs:
            self.lightgcn_enabled = False
            return

        loaded_scenes = 0
        for scene_dir in sorted(scene_dirs):
            scene_path = os.path.join(self.gcn_logs_dir, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            try:
                scene_id = int(scene_dir.split('_')[1])
            except (IndexError, ValueError):
                continue

            user_emb_path = os.path.join(scene_path, f"lightgcn_scene{scene_id}_user_emb_final.npy")
            item_emb_path = os.path.join(scene_path, f"lightgcn_scene{scene_id}_item_emb_final.npy")

            if os.path.exists(user_emb_path) and os.path.exists(item_emb_path):
                try:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    user_emb = torch.from_numpy(np.load(user_emb_path)).float().to(device)
                    item_emb = torch.from_numpy(np.load(item_emb_path)).float().to(device)

                    self.scene_user_embeddings_gpu[scene_id] = user_emb
                    self.scene_item_embeddings_gpu[scene_id] = item_emb

                    self.scene_avg_user_emb[scene_id] = user_emb.mean(dim=0)
                    self.scene_avg_item_emb[scene_id] = item_emb.mean(dim=0)

                    if self.graph_emb_dim == 0:
                        self.graph_emb_dim = user_emb.shape[1]

                    self._load_scene_mappings(scene_path, scene_id)
                    loaded_scenes += 1

                except Exception as e:
                    continue

        if loaded_scenes == 0:
            self.lightgcn_enabled = False
        else:
            self.embeddings_loaded_on_gpu = True
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_graph_embeddings_batch(self, user_ids, item_ids, scenes):
        """Batch optimized graph embedding retrieval"""
        batch_size = len(user_ids)
        device = next(self.parameters()).device

        if not self.lightgcn_enabled or self.graph_emb_dim == 0:
            return torch.randn(batch_size, self.expert_hidden_dim // 4, device=device) * 0.01

        graph_features = torch.zeros(batch_size, self.graph_emb_dim, device=device)

        scene_groups = {}
        for i, scene_id in enumerate(scenes):
            scene_id = scene_id.item() if hasattr(scene_id, 'item') else int(scene_id)
            if scene_id not in scene_groups:
                scene_groups[scene_id] = []
            scene_groups[scene_id].append(i)

        for scene_id, indices in scene_groups.items():
            if scene_id not in self.scene_user_embeddings_gpu:
                continue

            user_emb = self.scene_user_embeddings_gpu[scene_id]
            item_emb = self.scene_item_embeddings_gpu[scene_id]
            mappings = self.scene_mappings.get(scene_id, {})
            user_mapping = mappings.get('user_id_to_idx', {})
            item_mapping = mappings.get('item_id_to_idx', {})

            avg_user_emb = self.scene_avg_user_emb[scene_id]
            avg_item_emb = self.scene_avg_item_emb[scene_id]

            for i in indices:
                user_id = user_ids[i] if isinstance(user_ids[i], str) else str(user_ids[i])
                item_id = item_ids[i] if isinstance(item_ids[i], str) else str(item_ids[i])

                if user_id in user_mapping:
                    user_idx = user_mapping[user_id]
                    if user_idx < user_emb.shape[0]:
                        user_graph_emb = user_emb[user_idx]
                    else:
                        user_graph_emb = avg_user_emb
                else:
                    user_graph_emb = avg_user_emb

                if item_id in item_mapping:
                    item_idx = item_mapping[item_id]
                    if item_idx < item_emb.shape[0]:
                        item_graph_emb = item_emb[item_idx]
                    else:
                        item_graph_emb = avg_item_emb
                else:
                    item_graph_emb = avg_item_emb

                graph_features[i] = user_graph_emb * item_graph_emb

        return graph_features

    def forward(self, user_ids, item_ids, scenes):
        """Optimized forward pass"""
        if not self.lightgcn_enabled:
            batch_size = len(user_ids)
            device = next(self.parameters()).device
            light_features = torch.randn(batch_size, self.expert_hidden_dim // 4, device=device) * 0.01
            return self.graph_projection(light_features)

        graph_features = self.get_graph_embeddings_batch(user_ids, item_ids, scenes)
        output = self.graph_projection(graph_features)

        return output

    def _load_scene_mappings(self, scene_path, scene_id):
        """Load scene mappings for ID to index conversion"""
        mapping_files = {
            'user_id_to_idx': 'user_id_to_idx.pkl',
            'item_id_to_idx': 'item_id_to_idx.pkl'
        }

        scene_mappings = {}
        for mapping_name, mapping_file in mapping_files.items():
            mapping_path = os.path.join(scene_path, mapping_file)
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, 'rb') as f:
                        scene_mappings[mapping_name] = pickle.load(f)
                except Exception as e:
                    continue

        if scene_mappings:
            self.scene_mappings[scene_id] = scene_mappings


class WideMLPExpert(nn.Module):
    """Wide MLP Expert - Linear feature combination"""

    def __init__(self, total_input_dim=5024, expert_hidden_dim=512, dropout=0.2):
        super().__init__()
        self.expert_type = "wide_mlp"

        self.wide_layer = nn.Sequential(
            nn.Linear(total_input_dim, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_hidden_dim)
        )

    def forward(self, x):
        return self.wide_layer(x)


class DeepMLPExpert(nn.Module):
    """Deep MLP Expert - Deep nonlinear transformation"""

    def __init__(self, total_input_dim=5024, expert_hidden_dim=512, dropout=0.2):
        super().__init__()
        self.expert_type = "deep_mlp"

        self.deep_network = nn.Sequential(
            nn.Linear(total_input_dim, expert_hidden_dim * 2),
            nn.BatchNorm1d(expert_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(expert_hidden_dim * 2, expert_hidden_dim * 2),
            nn.BatchNorm1d(expert_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(expert_hidden_dim * 2, expert_hidden_dim),
            nn.BatchNorm1d(expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(expert_hidden_dim, expert_hidden_dim)
        )

    def forward(self, x):
        return self.deep_network(x)


class CrossMLPExpert(nn.Module):
    """Cross MLP Expert - Feature crossing network"""

    def __init__(self, total_input_dim=5024, expert_hidden_dim=512, dropout=0.2):
        super().__init__()
        self.expert_type = "cross_mlp"

        self.input_projection = nn.Linear(total_input_dim, expert_hidden_dim)

        self.cross_layers = nn.ModuleList([
            nn.Linear(expert_hidden_dim, expert_hidden_dim) for _ in range(3)
        ])

        self.deep_branch = nn.Sequential(
            nn.Linear(expert_hidden_dim, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_hidden_dim)
        )

        self.combination = nn.Sequential(
            nn.Linear(expert_hidden_dim * 2, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_hidden_dim)
        )

    def forward(self, x):
        x_proj = self.input_projection(x)

        x_cross = x_proj
        for cross_layer in self.cross_layers:
            x_cross = x_proj * cross_layer(x_cross) + x_cross

        x_deep = self.deep_branch(x_proj)

        combined = torch.cat([x_cross, x_deep], dim=1)
        output = self.combination(combined)

        return output


class HeterogeneousMMoE_FullEmbeddings(nn.Module):
    def __init__(self,
                 image_emb_dim=256,
                 text_emb_dim=3840,
                 id_emb_dim=928,
                 expert_hidden_dim=512,
                 tower_hidden_dims=[1024, 512, 256, 128],
                 output_dim=1,
                 dropout=0.3,
                 num_scenes=5,
                 use_lightgcn=True,
                 gcn_logs_dir="gcn_logs_incremental",
                 diversity_weight=0.01):

        super().__init__()

        self.num_scenes = num_scenes
        self.use_lightgcn = use_lightgcn
        self.diversity_weight = diversity_weight

        self.current_epoch = 0
        self.topk_config = {
            'warmup_epochs': 3,
            'balance_epochs': 5,
            'warmup_k': 5,
            'balance_k': 2,
            'final_k': 2,
            'warmup_max_weight': 0.60,
            'balance_max_weight_start': 0.65,
            'balance_max_weight_end': 0.75,
            'final_max_weight': 0.80,
            'warmup_temp': 2.0,
            'balance_temp_start': 1.5,
            'balance_temp_end': 1.2,
            'final_temp': 1.0,
            'warmup_alpha': 0.30,
            'balance_alpha_start': 0.20,
            'balance_alpha_end': 0.10,
            'final_alpha': 0.0
        }

        total_input_dim = image_emb_dim + text_emb_dim + id_emb_dim

        # Diversified heterogeneous expert networks
        self.cnn_image_expert = CNNImageExpert(
            image_dim=image_emb_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )

        self.transformer_text_expert = TransformerTextExpert(
            text_dim=text_emb_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )

        self.id_expert = IDExpert(
            id_dim=id_emb_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )

        if use_lightgcn:
            self.lightgcn_expert = LightGCNGraphExpert(
                expert_hidden_dim=expert_hidden_dim,
                dropout=dropout,
                gcn_logs_dir=gcn_logs_dir
            )

        self.wide_mlp_expert = WideMLPExpert(
            total_input_dim=total_input_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )

        self.deep_mlp_expert = DeepMLPExpert(
            total_input_dim=total_input_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )

        self.cross_mlp_expert = CrossMLPExpert(
            total_input_dim=total_input_dim,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )

        if use_lightgcn:
            self.num_experts = 7
        else:
            self.num_experts = 6

        # Top-K gating networks (one per scene)
        self.gates = nn.ModuleList([
            TopKGate(
                input_dim=total_input_dim,
                num_experts=self.num_experts,
                expert_hidden_dim=expert_hidden_dim,
                dropout=dropout
            ) for _ in range(num_scenes)
        ])

        # Prediction towers (one per scene)
        towers = []
        for _ in range(num_scenes):
            layers = []
            prev_dim = expert_hidden_dim

            for hidden_dim in tower_hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            towers.append(nn.Sequential(*layers))

        self.towers = nn.ModuleList(towers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model parameters"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_epoch(self, epoch):
        """Update current training epoch"""
        self.current_epoch = epoch

    def get_current_topk_params(self):
        """Get Top-K parameters based on current training stage"""
        config = self.topk_config
        epoch = self.current_epoch

        if epoch <= config['warmup_epochs']:
            # Warmup phase (0-3 epoch)
            return {
                'k': config['warmup_k'],
                'max_weight': config['warmup_max_weight'],
                'temperature': config['warmup_temp'],
                'alpha': config['warmup_alpha']
            }
        elif epoch <= config['balance_epochs']:
            # Balance phase (4-15 epoch) - linear interpolation
            progress = (epoch - config['warmup_epochs']) / (config['balance_epochs'] - config['warmup_epochs'])

            max_weight = config['balance_max_weight_start'] + progress * (
                    config['balance_max_weight_end'] - config['balance_max_weight_start']
            )
            temperature = config['balance_temp_start'] + progress * (
                    config['balance_temp_end'] - config['balance_temp_start']
            )
            alpha = config['balance_alpha_start'] + progress * (
                    config['balance_alpha_end'] - config['balance_alpha_start']
            )

            return {
                'k': config['balance_k'],
                'max_weight': max_weight,
                'temperature': temperature,
                'alpha': alpha
            }
        else:
            # Convergence phase (16+ epoch)
            return {
                'k': config['final_k'],
                'max_weight': config['final_max_weight'],
                'temperature': config['final_temp'],
                'alpha': config['final_alpha']
            }

    def forward(self, image_emb, text_emb, id_emb, scenes, user_ids=None, item_ids=None):

        batch_size = image_emb.size(0)

        # Diversified heterogeneous experts process features separately
        cnn_image_out = self.cnn_image_expert(image_emb)
        transformer_text_out = self.transformer_text_expert(text_emb)
        id_expert_out = self.id_expert(id_emb)

        combined_features = torch.cat([image_emb, text_emb, id_emb], dim=1)

        wide_mlp_out = self.wide_mlp_expert(combined_features)
        deep_mlp_out = self.deep_mlp_expert(combined_features)
        cross_mlp_out = self.cross_mlp_expert(combined_features)

        expert_outputs = [
            cnn_image_out,
            transformer_text_out,
            id_expert_out,
            wide_mlp_out,
            deep_mlp_out,
            cross_mlp_out
        ]

        if self.use_lightgcn and hasattr(self, 'lightgcn_expert'):
            if user_ids is not None and item_ids is not None:
                lightgcn_output = self.lightgcn_expert(user_ids, item_ids, scenes)
                expert_outputs.append(lightgcn_output)
            else:
                batch_size = image_emb.size(0)
                device = image_emb.device
                light_output = self.lightgcn_expert(
                    [f"default_user_{i}" for i in range(batch_size)],
                    [f"default_item_{i}" for i in range(batch_size)],
                    scenes
                )
                expert_outputs.append(light_output)

        expert_outputs = torch.stack(expert_outputs, dim=1)

        topk_params = self.get_current_topk_params()

        predictions = torch.zeros(batch_size, device=image_emb.device)
        total_diversity_loss = 0.0

        for scene_id in range(self.num_scenes):
            scene_mask = (scenes == scene_id)
            if not scene_mask.any():
                continue

            scene_features = combined_features[scene_mask]
            scene_expert_outputs = expert_outputs[scene_mask]

            gate_weights, selected_experts, diversity_loss = self.gates[scene_id](
                scene_features,
                k=topk_params['k'],
                max_weight=topk_params['max_weight'],
                temperature=topk_params['temperature'],
                alpha=topk_params['alpha'],
                training_epoch=self.current_epoch
            )

            total_diversity_loss += diversity_loss

            weighted_expert_output = (gate_weights.unsqueeze(-1) * scene_expert_outputs).sum(dim=1)
            scene_predictions = self.towers[scene_id](weighted_expert_output)

            predictions[scene_mask] = scene_predictions.squeeze(-1)

        avg_diversity_loss = total_diversity_loss / self.num_scenes

        return predictions, avg_diversity_loss

    def get_expert_weights_with_topk(self, image_emb, text_emb, id_emb, scenes, user_ids=None, item_ids=None):
        """Get expert weight distribution under Top-K mode for interpretability analysis"""
        with torch.no_grad():
            batch_size = image_emb.size(0)
            combined_features = torch.cat([image_emb, text_emb, id_emb], dim=1)
            topk_params = self.get_current_topk_params()

            scene_weights = {}
            expert_names = ['CNN_Image', 'Transformer_Text', 'ID_Expert', 'Wide_MLP', 'Deep_MLP', 'Cross_MLP']
            if self.use_lightgcn:
                expert_names.append('LightGCN_Graph')

            for scene_id in range(self.num_scenes):
                scene_mask = (scenes == scene_id)
                if scene_mask.any():
                    scene_features = combined_features[scene_mask]
                    gate_weights, selected_experts, _ = self.gates[scene_id](
                        scene_features,
                        k=topk_params['k'],
                        max_weight=topk_params['max_weight'],
                        temperature=topk_params['temperature'],
                        alpha=topk_params['alpha'],
                        training_epoch=self.current_epoch
                    )
                    gate_weights_np = gate_weights.cpu().numpy()
                    avg_weights = gate_weights_np.mean(axis=0)

                    topk_info = {}
                    if selected_experts is not None:
                        selected_experts_np = selected_experts.cpu().numpy()
                        expert_selection_count = np.bincount(selected_experts_np.flatten(), minlength=self.num_experts)
                        expert_selection_rate = expert_selection_count / (
                                    selected_experts_np.shape[0] * topk_params['k'])
                        topk_info = {
                            f'{expert_names[i]}_selection_rate': float(expert_selection_rate[i])
                            for i in range(len(expert_names))
                        }

                    scene_weights[f'scene_{scene_id}'] = {
                        **{expert_names[i]: float(avg_weights[i]) for i in range(len(expert_names))},
                        'topk_params': topk_params,
                        'topk_selection_stats': topk_info,
                        'weight_entropy': float(-np.sum(avg_weights * np.log(avg_weights + 1e-8))),
                        'max_weight': float(np.max(avg_weights)),
                        'weight_variance': float(np.var(avg_weights))
                    }

            return scene_weights

    def print_expert_analysis(self, data_loader, epoch=None, max_batches=5):
        """Print expert weight distribution analysis after training"""
        self.eval()

        all_weights = {f'scene_{i}': [] for i in range(self.num_scenes)}
        expert_names = ['CNN_Image', 'Transformer_Text', 'ID_Expert', 'Wide_MLP', 'Deep_MLP', 'Cross_MLP']
        if self.use_lightgcn:
            expert_names.append('LightGCN_Graph')

        batch_count = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                if batch_count >= max_batches:
                    break

                if len(batch) == 7:
                    image_emb, text_emb, id_emb, labels, scenes, user_ids, item_ids = batch
                else:
                    image_emb, text_emb, id_emb, labels, scenes = batch[:5]
                    user_ids, item_ids = None, None

                weights = self.get_expert_weights_with_topk(image_emb, text_emb, id_emb, scenes, user_ids, item_ids)

                for scene_key, scene_weights in weights.items():
                    all_weights[scene_key].append(scene_weights)

                total_samples += image_emb.size(0)
                batch_count += 1

        # Analysis output would go here
        return all_weights

    def get_training_phase_info(self):
        """Get current training phase information"""
        config = self.topk_config
        epoch = self.current_epoch

        if epoch <= config['warmup_epochs']:
            phase = "warmup"
            description = "multi-expert exploration, prevent early convergence"
        elif epoch <= config['balance_epochs']:
            phase = "balance"
            description = "gradual convergence, balance exploration and exploitation"
        else:
            phase = "convergence"
            description = "fine-tuning, optimal expert combination"

        current_params = self.get_current_topk_params()

        return {
            'current_epoch': epoch,
            'training_phase': phase,
            'phase_description': description,
            'current_topk_params': current_params,
            'config': config
        }


def create_heterogeneous_mmoe_model(**model_kwargs):

    default_kwargs = {
        'image_emb_dim': 256,
        'text_emb_dim': 3840,
        'id_emb_dim': 928,
        'expert_hidden_dim': 512,
        'tower_hidden_dims': [1024, 512, 256, 128],
        'output_dim': 1,
        'dropout': 0.3,
        'num_scenes': 5,
        'use_lightgcn': True,
        'gcn_logs_dir': "gcn_logs_incremental",
        'diversity_weight': 0.01
    }

    default_kwargs.update(model_kwargs)

    model = HeterogeneousMMoE_FullEmbeddings(**default_kwargs)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model

