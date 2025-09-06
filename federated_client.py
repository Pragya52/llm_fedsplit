"""
federated_client.py
Federated Learning Client Implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import copy

from dataclasses import dataclass

from model_architecture import ClientModel, SplitLLaMA3Model
from communication import ClientCommunicator, TensorSerializer

logger = logging.getLogger(__name__)

@dataclass
class ClientConfig:
    """Configuration for federated client"""
    client_id: int
    model_name: str = "meta-llama/Llama-3.2-1B"
    client_layers: int = 4
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    device: str = "auto"


class TextDataset(Dataset):
    """Custom dataset for text data"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        
        logger.info(f"Created dataset with {len(texts)} samples")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        # Set padding tokens to -100 so they're ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class FederatedClient:
    """
    Federated Learning Client
    Handles local training and communication with server
    """

    def __init__(self, config: ClientConfig, data: List[str], client_communicator: ClientCommunicator):
        self.config = config
        self.client_id = config.client_id
        self.communicator = client_communicator

        # Set device
        if config.device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        logger.info(f"Client {self.client_id} using device: {self.device}")
        
        # Initialize model
        model_handler = SplitLLaMA3Model(config.model_name, config.client_layers)
        self.model = model_handler.create_client_model().to(self.device)
        self.tokenizer = model_handler.tokenizer
        
        # Setup dataset and dataloader
        self.dataset = TextDataset(data, self.tokenizer, config.max_seq_length)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100
        )
        
        # Communication interface (will be set by main simulation)
        # self.communicator: Optional[ClientCommunicator] = None
        
        # Training metrics
        self.training_history = []
        self.current_round = 0
        
        logger.info(f"Client {self.client_id} initialized with {len(data)} samples")
    
    def train_batch(
        self, 
        batch: Dict[str, torch.Tensor],
        accumulation_step: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train on a single batch
        
        Args:
            batch: Batch of data
            accumulation_step: Current gradient accumulation step
            
        Returns:
            loss: Computed loss
            hidden_states: Output hidden states from client model
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass through client layers
        hidden_states, _ = self.model(input_ids, attention_mask)
        
        # Send hidden states to server and get gradients back
        if self.communicator is not None:
            server_response = self.communicator.process_batch(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                labels=labels,
                batch_id=accumulation_step
            )
        else:
            logger.error("Server is None\n")
            server_response = None
        
        if server_response is not None:
            # Get loss and gradients from server
            loss = server_response['loss']
            gradients = server_response['gradients']
            
            # Backward pass with server gradients
            hidden_states.backward(gradients)
        else:
            # Fallback: compute dummy loss locally (for testing)
            logger.warning(f"Client {self.client_id}: Server response failed, using dummy loss")
            loss = torch.nn.functional.mse_loss(
                hidden_states.mean(dim=1),
                torch.randn_like(hidden_states.mean(dim=1))
            )
            loss.backward()
        
        return loss, hidden_states
    
    def train_epoch(self) -> Dict:
        """
        Train for one epoch
        
        Returns:
            Epoch metrics
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Train on batch
            loss, _ = self.train_batch(
                batch, 
                accumulation_step=batch_idx % self.config.gradient_accumulation_steps
            )
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.debug(
                    f"Client {self.client_id} - Batch {batch_idx}/{len(self.dataloader)} - "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Final gradient update if needed
        if num_batches % self.config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Update scheduler
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            'loss': avg_loss,
            'num_batches': num_batches,
            'num_samples': len(self.dataset),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def local_training(self, epochs: int) -> Dict:
        """
        Perform local training for specified epochs
        
        Args:
            epochs: Number of epochs to train
            
        Returns:
            Training results including model state and metrics
        """
        logger.info(f"Client {self.client_id} starting local training for {epochs} epochs")
        
        epoch_metrics = []
        
        for epoch in range(epochs):
            metrics = self.train_epoch()
            epoch_metrics.append(metrics)
            
            logger.info(
                f"Client {self.client_id} - Epoch {epoch+1}/{epochs} - "
                f"Loss: {metrics['loss']:.4f} - LR: {metrics['lr']:.6f}"
            )
        
        # Store training history
        self.training_history.extend(epoch_metrics)
        
        return {
            'client_id': self.client_id,
            'round': self.current_round,
            'metrics': epoch_metrics,
            'model_state': self.get_model_state(),
            'num_samples': len(self.dataset)
        }
    
    def get_model_state(self) -> Dict:
        """Get current model state dict"""
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_state(self, state_dict: Dict):
        """
        Update model with new state dict
        
        Args:
            state_dict: New model parameters
        """
        self.model.load_state_dict(state_dict)
        logger.debug(f"Client {self.client_id} model updated with new parameters")
    
    def sync_with_server(self, round_number: int) -> bool:
        """
        Synchronize model with server's global model
        
        Args:
            round_number: Current federated round
            
        Returns:
            Success status
        """
        self.current_round = round_number
        
        logger.info(f"Client {self.client_id} syncing with server for round {round_number}")
        
        # Get global model from server
        if self.communicator is not None:
            global_state = self.communicator.get_global_model(round_number)
        else:
            logger.error(f"Client {self.client_id}: No communicator available")
            return False
        
        if global_state is not None:
            self.set_model_state(global_state)
            return True
        else:
            logger.error(f"Client {self.client_id} failed to sync with server")
            return False
    
    def send_update_to_server(self, training_results: Dict) -> bool:
        """
        Send local training results to server
        
        Args:
            training_results: Results from local training
            
        Returns:
            Success status
        """
        logger.info(f"Client {self.client_id} sending update to server")
        
        if self.communicator is None:
            logger.error(f"Client {self.client_id}: No communicator available")
            return False
        
        success = self.communicator.send_update(
            model_state=training_results['model_state'],
            metrics=training_results['metrics'],
            num_samples=training_results['num_samples'],
            round_number=self.current_round
        )
        
        return success
    
    def participate_in_round(self, round_number: int, local_epochs: int) -> bool:
        """
        Participate in a federated learning round
        
        Args:
            round_number: Current round number
            local_epochs: Number of local training epochs
            
        Returns:
            Success status
        """
        try:
            # Sync with global model
            if not self.sync_with_server(round_number):
                return False
            
            # Perform local training
            training_results = self.local_training(local_epochs)
            
            # Send updates to server
            if not self.send_update_to_server(training_results):
                return False
            
            logger.info(f"Client {self.client_id} completed round {round_number}")
            return True
            
        except Exception as e:
            logger.error(f"Client {self.client_id} error in round {round_number}: {str(e)}")
            return False
    
    def save_checkpoint(self, filepath: str):
        """Save client checkpoint"""
        checkpoint = {
            'client_id': self.client_id,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'current_round': self.current_round
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Client {self.client_id} checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load client checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint['training_history']
        self.current_round = checkpoint['current_round']
        
        logger.info(f"Client {self.client_id} checkpoint loaded from {filepath}")
    
    def close(self):
        """Cleanup and close connections"""
        # No explicit cleanup needed for communication hub
        logger.info(f"Client {self.client_id} closed")
