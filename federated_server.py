"""
federated_server.py
Federated Learning Server Implementation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import threading
import time
from dataclasses import dataclass
import copy
from collections import defaultdict

from model_architecture import ServerModel, ClientModel, SplitLLaMA3Model

logger = logging.getLogger(__name__)

@dataclass
class ServerConfig:
    """Configuration for federated server"""
    model_name: str = "meta-llama/Llama-3.2-1B"
    client_layers: int = 4
    aggregation_strategy: str = "fedavg"  # fedavg, weighted, or median
    min_clients_per_round: int = 2
    max_rounds: int = 100
    checkpoint_frequency: int = 5
    device: str = "auto"


class FederatedAggregator:
    """
    Handles different aggregation strategies for federated learning
    """
    
    @staticmethod
    def fedavg(client_updates: List[Dict]) -> Dict:
        """
        Federated Averaging (FedAvg) - Simple averaging of model parameters
        
        Args:
            client_updates: List of client model updates
            
        Returns:
            Aggregated model state
        """
        if not client_updates:
            return {}
        
        # Get first client's state dict structure
        first_state = client_updates[0]['model_state']
        aggregated_state = {}
        
        for key in first_state.keys():
            # Stack all client parameters for this key
            stacked_params = torch.stack([
                update['model_state'][key] for update in client_updates
            ])
            
            # Simple average
            aggregated_state[key] = stacked_params.mean(dim=0)
        
        return aggregated_state
    
    @staticmethod
    def weighted_average(client_updates: List[Dict]) -> Dict:
        """
        Weighted averaging based on number of samples
        
        Args:
            client_updates: List of client model updates with num_samples
            
        Returns:
            Weighted aggregated model state
        """
        if not client_updates:
            return {}
        
        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        if total_samples == 0:
            # Fall back to simple averaging
            return FederatedAggregator.fedavg(client_updates)
        
        first_state = client_updates[0]['model_state']
        aggregated_state = {}
        
        for key in first_state.keys():
            # Weighted sum
            weighted_sum = torch.zeros_like(first_state[key])
            
            for update in client_updates:
                weight = update['num_samples'] / total_samples
                weighted_sum += update['model_state'][key] * weight
            
            aggregated_state[key] = weighted_sum
        
        return aggregated_state
    
    @staticmethod
    def median_aggregation(client_updates: List[Dict]) -> Dict:
        """
        Median aggregation for robustness against outliers
        
        Args:
            client_updates: List of client model updates
            
        Returns:
            Median aggregated model state
        """
        if not client_updates:
            return {}
        
        first_state = client_updates[0]['model_state']
        aggregated_state = {}
        
        for key in first_state.keys():
            # Stack all client parameters
            stacked_params = torch.stack([
                update['model_state'][key] for update in client_updates
            ])
            
            # Compute median
            aggregated_state[key] = torch.median(stacked_params, dim=0)[0]
        
        return aggregated_state


class FederatedServer:
    """
    Federated Learning Server
    Manages global model, aggregation, and coordination
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        
        # Set device
        if config.device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        logger.info(f"Server using device: {self.device}")
        
        # Initialize models
        model_handler = SplitLLaMA3Model(config.model_name, config.client_layers)
        
        # Server model (for processing hidden states)
        self.server_model = model_handler.create_server_model().to(self.device)
        
        # Global client model (for aggregation)
        self.global_client_model = model_handler.create_client_model().to(self.device)
        
        # Tokenizer for server-side processing
        self.tokenizer = model_handler.tokenizer
        
        # Aggregator
        self.aggregator = FederatedAggregator()
        
        # Round management
        self.current_round = 0
        self.round_lock = threading.Lock()
        self.client_updates = defaultdict(list)  # round -> list of updates
        self.round_metrics = []
        
        # Client tracking
        self.registered_clients = set()
        self.active_clients = set()
        
        logger.info(f"Federated server initialized with {config.aggregation_strategy} aggregation")
    
    def register_client(self, client_id: int) -> bool:
        """
        Register a new client
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            Success status
        """
        with self.round_lock:
            self.registered_clients.add(client_id)
            logger.info(f"Client {client_id} registered. Total clients: {len(self.registered_clients)}")
        return True
    
    def start_round(self) -> Dict:
        """
        Start a new federated learning round
        
        Returns:
            Round information and global model state
        """
        with self.round_lock:
            self.current_round += 1
            self.client_updates[self.current_round] = []
            self.active_clients.clear()
            
            logger.info(f"Starting round {self.current_round}")
            
            return {
                'round': self.current_round,
                'global_model_state': self.global_client_model.state_dict()
            }
    
    def process_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict:
        """
        Process hidden states through server model
        
        Args:
            hidden_states: Hidden states from client
            attention_mask: Attention mask
            labels: Ground truth labels
            
        Returns:
            Loss and gradients for backpropagation
        """
        # Ensure tensors are on correct device
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
        labels = labels.to(self.device)
        
        # Enable gradient computation for hidden states
        hidden_states.requires_grad_(True)
        
        # Forward pass through server layers
        logits, _, _ = self.server_model(
            hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        # Compute loss
        loss = self.server_model.compute_loss(logits, labels)
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Get gradients for hidden states
        gradients = hidden_states.grad.clone()
        
        # Clear gradients
        hidden_states.grad = None
        
        return {
            'loss': loss.detach(),
            'gradients': gradients.detach(),
            'logits': logits.detach()
        }
    
    def receive_client_update(self, client_update: Dict) -> bool:
        """
        Receive and store client update
        
        Args:
            client_update: Update from client containing model state and metrics
            
        Returns:
            Success status
        """
        client_id = client_update['client_id']
        round_num = client_update.get('round', self.current_round)
        
        with self.round_lock:
            # Check if update is for current round
            if round_num != self.current_round:
                logger.warning(
                    f"Client {client_id} sent update for round {round_num}, "
                    f"but current round is {self.current_round}"
                )
                return False
            
            # Store update
            self.client_updates[round_num].append(client_update)
            self.active_clients.add(client_id)
            
            logger.info(
                f"Received update from client {client_id} for round {round_num}. "
                f"Active clients: {len(self.active_clients)}/{len(self.registered_clients)}"
            )
            
            return True
    
    def aggregate_client_models(self, client_updates: List[Dict]) -> Dict:
        """
        Aggregate client models using configured strategy
        
        Args:
            client_updates: List of client updates
            
        Returns:
            Aggregated model state
        """
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return self.global_client_model.state_dict()
        
        logger.info(f"Aggregating {len(client_updates)} client updates using {self.config.aggregation_strategy}")
        
        # Select aggregation strategy
        if self.config.aggregation_strategy == "fedavg":
            aggregated_state = self.aggregator.fedavg(client_updates)
        elif self.config.aggregation_strategy == "weighted":
            aggregated_state = self.aggregator.weighted_average(client_updates)
        elif self.config.aggregation_strategy == "median":
            aggregated_state = self.aggregator.median_aggregation(client_updates)
        else:
            logger.error(f"Unknown aggregation strategy: {self.config.aggregation_strategy}")
            aggregated_state = self.aggregator.fedavg(client_updates)
        
        return aggregated_state
    
    def end_round(self) -> Dict:
        """
        End current round and perform aggregation
        
        Returns:
            Round summary with metrics and aggregated model
        """
        with self.round_lock:
            round_num = self.current_round
            client_updates = self.client_updates[round_num]
            
            logger.info(
                f"Ending round {round_num} with {len(client_updates)} updates "
                f"from {len(self.active_clients)} clients"
            )
            
            if len(client_updates) < self.config.min_clients_per_round:
                logger.warning(
                    f"Insufficient clients: {len(client_updates)} < {self.config.min_clients_per_round}"
                )
            
            # Aggregate client models
            aggregated_state = self.aggregate_client_models(client_updates)
            
            # Update global model
            self.global_client_model.load_state_dict(aggregated_state)
            
            # Calculate metrics
            avg_loss = 0
            total_samples = 0
            
            if client_updates:
                losses = []
                for update in client_updates:
                    if 'metrics' in update and update['metrics']:
                        # Average loss across epochs for this client
                        client_losses = [m['loss'] for m in update['metrics']]
                        losses.append(np.mean(client_losses))
                        total_samples += update.get('num_samples', 0)
                
                avg_loss = np.mean(losses) if losses else 0
            
            round_summary = {
                'round': round_num,
                'num_clients': len(client_updates),
                'active_clients': list(self.active_clients),
                'avg_loss': avg_loss,
                'total_samples': total_samples,
                'global_model_state': aggregated_state
            }
            
            # Store metrics
            self.round_metrics.append({
                'round': round_num,
                'num_clients': len(client_updates),
                'avg_loss': avg_loss,
                'total_samples': total_samples
            })
            
            # Checkpoint if needed
            if round_num % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(f"federated_checkpoint_round_{round_num}.pt")
            
            return round_summary
    
    def get_global_model_state(self) -> Dict:
        """Get current global model state"""
        return copy.deepcopy(self.global_client_model.state_dict())
    
    def set_global_model_state(self, state_dict: Dict):
        """Set global model state"""
        self.global_client_model.load_state_dict(state_dict)
        logger.info("Global model updated")
    
    def evaluate_global_model(self, test_data: Optional[List[str]] = None) -> Dict:
        """
        Evaluate the global model
        
        Args:
            test_data: Optional test data for evaluation
            
        Returns:
            Evaluation metrics
        """
        self.global_client_model.eval()
        self.server_model.eval()
        
        if test_data is None:
            # Use dummy data for testing
            test_data = [
                "The future of AI is",
                "Federated learning enables",
                "Large language models can"
            ]
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in test_data:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                # Forward pass through client model
                hidden_states, _ = self.global_client_model(
                    encoding['input_ids'],
                    encoding['attention_mask']
                )
                
                # Forward pass through server model
                logits, _, _ = self.server_model(
                    hidden_states,
                    encoding['attention_mask']
                )
                
                # Compute perplexity (simplified)
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    encoding['input_ids'].view(-1)
                )
                
                total_loss += loss.item() * encoding['input_ids'].numel()
                total_tokens += encoding['input_ids'].numel()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.global_client_model.train()
        self.server_model.train()
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'num_samples': len(test_data)
        }
    
    def get_training_summary(self) -> Dict:
        """Get summary of training progress"""
        if not self.round_metrics:
            return {}
        
        return {
            'total_rounds': len(self.round_metrics),
            'current_round': self.current_round,
            'registered_clients': len(self.registered_clients),
            'average_clients_per_round': np.mean([m['num_clients'] for m in self.round_metrics]),
            'average_loss': np.mean([m['avg_loss'] for m in self.round_metrics]),
            'total_samples_processed': sum(m['total_samples'] for m in self.round_metrics),
            'round_history': self.round_metrics[-10:]  # Last 10 rounds
        }
    
    def save_checkpoint(self, filepath: str):
        """Save server checkpoint"""
        checkpoint = {
            'current_round': self.current_round,
            'global_client_model_state': self.global_client_model.state_dict(),
            'server_model_state': self.server_model.state_dict(),
            'round_metrics': self.round_metrics,
            'registered_clients': list(self.registered_clients),
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Server checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load server checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.current_round = checkpoint['current_round']
        self.global_client_model.load_state_dict(checkpoint['global_client_model_state'])
        self.server_model.load_state_dict(checkpoint['server_model_state'])
        self.round_metrics = checkpoint['round_metrics']
        self.registered_clients = set(checkpoint['registered_clients'])
        
        logger.info(f"Server checkpoint loaded from {filepath}")
        logger.info(f"Resumed at round {self.current_round} with {len(self.registered_clients)} clients")
