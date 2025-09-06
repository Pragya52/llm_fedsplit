"""
communication.py
Simple Communication Layer for Federated Learning using Function Calls
"""

import torch
import pickle
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import copy
from dataclasses import dataclass
import queue
import threading

logger = logging.getLogger(__name__)

# ==================== Serialization Utilities ====================
class TensorSerializer:
    """Utility class for serializing/deserializing PyTorch tensors and objects"""
    
    @staticmethod
    def serialize_tensor(tensor: torch.Tensor) -> bytes:
        """Serialize a PyTorch tensor to bytes"""
        try:
            np_array = tensor.detach().cpu().numpy()
            return pickle.dumps(np_array)
        except Exception as e:
            logger.error(f"Error serializing tensor: {e}")
            raise
    
    @staticmethod
    def deserialize_tensor(data: bytes, device: str = 'cpu') -> torch.Tensor:
        """Deserialize bytes to PyTorch tensor"""
        try:
            np_array = pickle.loads(data)
            tensor = torch.from_numpy(np_array)
            return tensor.to(device)
        except Exception as e:
            logger.error(f"Error deserializing tensor: {e}")
            raise
    
    @staticmethod
    def serialize_dict(data: Dict) -> bytes:
        """Serialize a dictionary to bytes"""
        return pickle.dumps(data)
    
    @staticmethod
    def deserialize_dict(data: bytes) -> Dict:
        """Deserialize bytes to dictionary"""
        return pickle.loads(data)
    
    @staticmethod
    def deep_copy_state(state_dict: Dict) -> Dict:
        """Create a deep copy of model state dict"""
        return copy.deepcopy(state_dict)


# ==================== Communication Hub ====================
class CommunicationHub:
    """
    Central communication hub that simulates network communication
    using direct function calls with optional latency simulation
    """
    
    def __init__(self, simulate_latency: bool = False):
        self.simulate_latency = simulate_latency
        self.server = None
        self.clients = {}
        self.message_queue = queue.Queue()
        self.lock = threading.Lock()
        
        logger.info("Communication hub initialized")
    
    def register_server(self, server):
        """Register the federated server"""
        self.server = server
        logger.info("Server registered with communication hub")
    
    def register_client(self, client_id: int, client):
        """Register a federated client"""
        with self.lock:
            self.clients[client_id] = client
            if self.server:
                self.server.register_client(client_id)
        logger.info(f"Client {client_id} registered with communication hub")
    
    def _simulate_network_delay(self):
        """Simulate network latency if enabled"""
        if self.simulate_latency:
            delay = np.random.uniform(0.01, 0.1)  # 10-100ms
            import time
            time.sleep(delay)
    
    # Server -> Client Communications
    def broadcast_round_start(self, round_number: int) -> Dict:
        """Server broadcasts round start to all clients"""
        self._simulate_network_delay()
        
        if not self.server:
            raise RuntimeError("Server not registered")
        
        round_info = self.server.start_round()
        
        # Notify all registered clients
        for client_id in self.clients:
            logger.debug(f"Notifying client {client_id} of round {round_number} start")
        
        return round_info
    
    def get_global_model(self, client_id: int, round_number: int) -> Optional[Dict]:
        """Client requests global model from server"""
        self._simulate_network_delay()
        
        if not self.server:
            logger.error("Server not registered")
            return None
        
        try:
            global_state = self.server.get_global_model_state()
            # Deep copy to simulate network transfer
            return TensorSerializer.deep_copy_state(global_state)
        except Exception as e:
            logger.error(f"Error getting global model for client {client_id}: {e}")
            return None
    
    # Client -> Server Communications
    def send_client_update(
        self, 
        client_id: int, 
        model_state: Dict,
        metrics: List[Dict],
        num_samples: int,
        round_number: int
    ) -> bool:
        """Client sends update to server"""
        self._simulate_network_delay()
        
        if not self.server:
            logger.error("Server not registered")
            return False
        
        try:
            # Deep copy to simulate network transfer
            client_update = {
                'client_id': client_id,
                'model_state': TensorSerializer.deep_copy_state(model_state),
                'metrics': copy.deepcopy(metrics),
                'num_samples': num_samples,
                'round': round_number
            }
            
            return self.server.receive_client_update(client_update)
        except Exception as e:
            logger.error(f"Error sending update from client {client_id}: {e}")
            return False
    
    def process_hidden_states(
        self,
        client_id: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        batch_id: int = 0
    ) -> Optional[Dict]:
        """Process hidden states through server model"""
        self._simulate_network_delay()
        
        if not self.server:
            logger.error("Server not registered")
            return None
        
        try:
            # Clone tensors to simulate network transfer
            hidden_states_copy = hidden_states.clone().detach()
            attention_mask_copy = attention_mask.clone().detach() if attention_mask is not None else None
            labels_copy = labels.clone().detach()
            
            result = self.server.process_hidden_states(
                hidden_states_copy,
                attention_mask_copy,
                labels_copy
            )
            
            # Clone results to simulate network transfer back
            return {
                'loss': result['loss'].clone().detach(),
                'gradients': result['gradients'].clone().detach()
            }
        except Exception as e:
            logger.error(f"Error processing hidden states for client {client_id}: {e}")
            return None
    
    # Aggregation trigger
    def trigger_aggregation(self, round_number: int) -> Dict:
        """Trigger aggregation on server"""
        self._simulate_network_delay()
        
        if not self.server:
            raise RuntimeError("Server not registered")
        
        return self.server.end_round()
    
    # Utility methods
    def get_registered_clients(self) -> List[int]:
        """Get list of registered client IDs"""
        with self.lock:
            return list(self.clients.keys())
    
    def get_active_clients(self) -> List[int]:
        """Get list of active client IDs"""
        if self.server:
            return list(self.server.active_clients)
        return []
    
    def reset(self):
        """Reset communication hub"""
        with self.lock:
            self.server = None
            self.clients.clear()
            while not self.message_queue.empty():
                self.message_queue.get()
        logger.info("Communication hub reset")


# ==================== Client Communication Interface ====================
class ClientCommunicator:
    """
    Client-side communication interface
    Handles all communication between client and server through the hub
    """
    
    def __init__(self, client_id: int, communication_hub: CommunicationHub):
        self.client_id = client_id
        self.hub = communication_hub
        self.current_round = 0
        
        logger.info(f"Client {client_id} communicator initialized")
    
    def get_global_model(self, round_number: int) -> Optional[Dict]:
        """Request global model from server"""
        logger.debug(f"Client {self.client_id} requesting global model for round {round_number}")
        return self.hub.get_global_model(self.client_id, round_number)
    
    def send_update(
        self,
        model_state: Dict,
        metrics: List[Dict],
        num_samples: int,
        round_number: int
    ) -> bool:
        """Send local update to server"""
        logger.debug(f"Client {self.client_id} sending update for round {round_number}")
        return self.hub.send_client_update(
            self.client_id,
            model_state,
            metrics,
            num_samples,
            round_number
        )
    
    def process_batch(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        batch_id: int = 0
    ) -> Optional[Dict]:
        """Send hidden states to server for processing"""
        return self.hub.process_hidden_states(
            self.client_id,
            hidden_states,
            attention_mask,
            labels,
            batch_id
        )


# ==================== Server Communication Interface ====================
class ServerCommunicator:
    """
    Server-side communication interface
    Handles broadcasting and client management through the hub
    """
    
    def __init__(self, communication_hub: CommunicationHub):
        self.hub = communication_hub
        self.current_round = 0
        
        logger.info({"communication/server_communicator_initialized": True})
    
    def broadcast_new_round(self, round_number: int) -> Dict:
        """Broadcast new round to all clients"""
        self.current_round = round_number
        logger.info({"communication/broadcasting_round": round_number})
        return self.hub.broadcast_round_start(round_number)
    
    def wait_for_clients(self, min_clients: int, timeout: float = 300) -> bool:
        """
        Wait for minimum number of clients to join
        
        Args:
            min_clients: Minimum number of clients required
            timeout: Maximum wait time in seconds
            
        Returns:
            True if minimum clients joined, False if timeout
        """
        import time
        start_time = time.time()
        
        while len(self.hub.get_registered_clients()) < min_clients:
            if time.time() - start_time > timeout:
                logger.info({
                    "communication/timeout_waiting_for_clients": True,
                    "communication/clients_joined": len(self.hub.get_registered_clients()),
                    "communication/min_clients_required": min_clients
                })
                return False
            time.sleep(1)
        
        logger.info({
            "communication/min_clients_met": True,
            "communication/total_clients": len(self.hub.get_registered_clients())
        })
        return True
    
    def trigger_aggregation(self) -> Dict:
        """Trigger aggregation after all clients have sent updates"""
        logger.info({"communication/triggering_aggregation": self.current_round})
        return self.hub.trigger_aggregation(self.current_round)
    
    def get_active_clients(self) -> List[int]:
        """Get list of clients that have sent updates this round"""
        return self.hub.get_active_clients()
