"""
main.py
Main Federated Learning Simulation Script
"""

import torch
import numpy as np
import logging
from typing import List, Dict
import argparse
import json
from dataclasses import dataclass, asdict
import time
import os
# HuggingFace datasets
from datasets import load_dataset

# Import our modules
from model_architecture import SplitLLaMA3Model
from federated_client import FederatedClient, ClientConfig
from federated_server import FederatedServer, ServerConfig
from communication import CommunicationHub, ClientCommunicator, ServerCommunicator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration for federated learning simulation"""
    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    client_layers: int = 4
    
    # Federation configuration
    num_clients: int = 3
    num_rounds: int = 10
    local_epochs: int = 2
    min_clients_per_round: int = 2
    
    # Training configuration
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_seq_length: int = 256
    
    # System configuration
    device: str = "auto"
    checkpoint_dir: str = "./checkpoints"
    simulate_latency: bool = False
    aggregation_strategy: str = "fedavg"  # fedavg, weighted, or median
    
    # Data configuration
    samples_per_client_min: int = 50
    samples_per_client_max: int = 100000
    data_distribution: str = "iid"  # iid or non-iid


class DataGenerator:
    """Generate synthetic data for federated learning simulation"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def load_and_split_hf_data(self) -> List[List[str]]:
        """
        Load data from HuggingFace 'bigbio/pubmed_qa' and split IID among clients
        Returns: List of lists, each for a client
        """
        logger.info("Loading HuggingFace dataset: bigbio/pubmed_qa")
        cache_dir = os.path.join(os.getcwd(), "hf_cache")
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", cache_dir=cache_dir)
        data_split = dataset["train"]  # Use the correct split
        # Use 'question' field for text, fallback to 'context' if needed

        all_texts = []
        for item in data_split:
            if 'question' in item and item['question']:
                all_texts.append(item['question'])
            elif 'context' in item and item['context']:
                all_texts.append(item['context'])
        logger.info(f"Loaded {len(all_texts)} samples from bigbio/pubmed_qa")
        # Shuffle and split IID
        np.random.shuffle(all_texts)
        num_clients = self.config.num_clients
        samples_per_client = len(all_texts) // num_clients
        client_data = [
            all_texts[i*samples_per_client:(i+1)*samples_per_client]
            for i in range(num_clients)
        ]
        # Add remaining samples to last client
        if len(all_texts) % num_clients != 0:
            client_data[-1].extend(all_texts[num_clients*samples_per_client:])
        return client_data
    
    def _get_client_templates(self, client_id: int) -> List[str]:
        """Get subset of templates for non-IID distribution"""
        # Each client gets 60% of templates with some overlap
        num_templates = int(len(self.templates) * 0.6)
        start_idx = (client_id * 3) % len(self.templates)
        
        client_templates = []
        for i in range(num_templates):
            idx = (start_idx + i) % len(self.templates)
            client_templates.append(self.templates[idx])
        
        return client_templates

class FederatedLearningSimulation:
    """
    Main federated learning simulation orchestrator
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize communication hub
        self.hub = CommunicationHub(simulate_latency=config.simulate_latency)
        
        # Initialize server
        server_config = ServerConfig(
            model_name=config.model_name,
            client_layers=config.client_layers,
            aggregation_strategy=config.aggregation_strategy,
            min_clients_per_round=config.min_clients_per_round,
            device=config.device
        )
        self.server = FederatedServer(server_config)
        self.server_comm = ServerCommunicator(self.hub)
        self.hub.register_server(self.server)
        
        # Data generator
        self.data_generator = DataGenerator(config)
        
        # Initialize clients
        self.clients = []
        self.client_comms = []
        self._initialize_clients()
        
        # Metrics tracking
        self.simulation_metrics = {
            'rounds': [],
            'client_metrics': {},
            'server_metrics': []
        }
        
        logger.info(f"Simulation initialized with {len(self.clients)} clients")
    
    def _initialize_clients(self):
        """Initialize federated clients with HF data"""
        # Load and split HF data IID
        client_data_list = self.data_generator.load_and_split_hf_data()
        for i in range(self.config.num_clients):
            client_data = client_data_list[i]
            client_config = ClientConfig(
                client_id=i,
                model_name=self.config.model_name,
                client_layers=self.config.client_layers,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                max_seq_length=self.config.max_seq_length,
                device=self.config.device
            )
            client_comm = ClientCommunicator(i, self.hub)
            client = FederatedClient(client_config, client_data, client_comm)
            client.communicator = client_comm
            self.hub.register_client(i, client)
            self.clients.append(client)
            self.client_comms.append(client_comm)
    
    def run_round(self, round_number: int) -> Dict:
        """
        Run a single federated learning round
        
        Args:
            round_number: Current round number
            
        Returns:
            Round summary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {round_number}/{self.config.num_rounds}")
        logger.info(f"{'='*60}")
        
        # Start new round on server
        round_info = self.server_comm.broadcast_new_round(round_number)
        global_state = round_info['global_model_state']
        
        # Track round metrics
        round_metrics = {
            'round': round_number,
            'client_updates': [],
            'start_time': time.time()
        }
        
        # Each client performs local training
        for i, (client, client_comm) in enumerate(zip(self.clients, self.client_comms)):
            logger.info(f"\nClient {i} starting local training...")
            
            # Get global model
            client.set_model_state(global_state)
            
            # Local training
            training_results = client.local_training(self.config.local_epochs)
            
            # Send update to server
            success = client_comm.send_update(
                model_state=training_results['model_state'],
                metrics=training_results['metrics'],
                num_samples=training_results['num_samples'],
                round_number=round_number
            )
            
            if success:
                round_metrics['client_updates'].append({
                    'client_id': i,
                    'loss': np.mean([m['loss'] for m in training_results['metrics']]),
                    'num_samples': training_results['num_samples']
                })
                logger.info(f"Client {i} successfully sent update")
            else:
                logger.error(f"Client {i} failed to send update")
        
        # Server aggregation
        logger.info("\nServer performing aggregation...")
        aggregation_result = self.server_comm.trigger_aggregation()
        
        # Record metrics
        round_metrics['end_time'] = time.time()
        round_metrics['duration'] = round_metrics['end_time'] - round_metrics['start_time']
        round_metrics['avg_loss'] = aggregation_result['avg_loss']
        round_metrics['num_clients'] = aggregation_result['num_clients']
        
        # Evaluate global model
        eval_metrics = self.server.evaluate_global_model()
        round_metrics['eval_loss'] = eval_metrics['loss']
        round_metrics['eval_perplexity'] = eval_metrics['perplexity']
        
        logger.info(f"\nRound {round_number} Summary:")
        logger.info(f"  - Duration: {round_metrics['duration']:.2f}s")
        logger.info(f"  - Active clients: {round_metrics['num_clients']}")
        logger.info(f"  - Average training loss: {round_metrics['avg_loss']:.4f}")
        logger.info(f"  - Evaluation loss: {round_metrics['eval_loss']:.4f}")
        logger.info(f"  - Perplexity: {round_metrics['eval_perplexity']:.2f}")
        
        return round_metrics
    
    def run(self):
        """Run the complete federated learning simulation"""
        logger.info("\n" + "="*60)
        logger.info("STARTING FEDERATED LEARNING SIMULATION")
        logger.info("="*60)
        logger.info(f"Configuration: {json.dumps(asdict(self.config), indent=2)}")
        
        # Wait for minimum clients
        if not self.server_comm.wait_for_clients(self.config.min_clients_per_round, timeout=10):
            logger.error("Failed to get minimum clients")
            return
        
        # Run federated rounds
        for round_num in range(1, self.config.num_rounds + 1):
            round_metrics = self.run_round(round_num)
            self.simulation_metrics['rounds'].append(round_metrics)
            
            # Save checkpoint every 5 rounds
            if round_num % 5 == 0:
                self.save_checkpoint(round_num)
        
        # Final evaluation
        logger.info("\n" + "="*60)
        logger.info("SIMULATION COMPLETED")
        logger.info("="*60)
        
        self.print_final_summary()
        self.save_results()
    
    def print_final_summary(self):
        """Print final simulation summary"""
        rounds = self.simulation_metrics['rounds']
        
        if not rounds:
            logger.warning("No rounds completed")
            return
        
        # Calculate summary statistics
        avg_duration = np.mean([r['duration'] for r in rounds])
        avg_loss = np.mean([r['avg_loss'] for r in rounds])
        final_loss = rounds[-1]['eval_loss']
        final_perplexity = rounds[-1]['eval_perplexity']
        
        # Loss improvement
        initial_loss = rounds[0]['avg_loss'] if rounds else 0
        loss_improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
        
        logger.info("\nFinal Summary:")
        logger.info(f"  Total rounds: {len(rounds)}")
        logger.info(f"  Average round duration: {avg_duration:.2f}s")
        logger.info(f"  Average training loss: {avg_loss:.4f}")
        logger.info(f"  Final evaluation loss: {final_loss:.4f}")
        logger.info(f"  Final perplexity: {final_perplexity:.2f}")
        logger.info(f"  Loss improvement: {loss_improvement:.2f}%")
        
        # Training summary from server
        training_summary = self.server.get_training_summary()
        logger.info(f"\nServer Statistics:")
        logger.info(f"  Total samples processed: {training_summary.get('total_samples_processed', 0)}")
        logger.info(f"  Average clients per round: {training_summary.get('average_clients_per_round', 0):.2f}")
    
    def save_checkpoint(self, round_num: int):
        """Save simulation checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_round_{round_num}.pt"
        )
        
        # Save server checkpoint
        self.server.save_checkpoint(checkpoint_path)
        
        # Save simulation metrics
        metrics_path = os.path.join(
            self.config.checkpoint_dir,
            f"metrics_round_{round_num}.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(self.simulation_metrics, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved for round {round_num}")
    
    def save_results(self):
        """Save final results"""
        # Save final model
        final_model_path = os.path.join(
            self.config.checkpoint_dir,
            "final_global_model.pt"
        )
        torch.save(self.server.global_client_model.state_dict(), final_model_path)
        
        # Save metrics
        metrics_path = os.path.join(
            self.config.checkpoint_dir,
            "final_metrics.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(self.simulation_metrics, f, indent=2, default=str)
        
        logger.info(f"Final results saved to {self.config.checkpoint_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                      help="Model name from HuggingFace")
    parser.add_argument("--client_layers", type=int, default=4,
                      help="Number of layers on client")
    
    # Federation arguments
    parser.add_argument("--num_clients", type=int, default=3,
                      help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=10,
                      help="Number of federated rounds")
    parser.add_argument("--local_epochs", type=int, default=2,
                      help="Number of local training epochs")
    parser.add_argument("--aggregation", type=str, default="fedavg",
                      choices=["fedavg", "weighted", "median"],
                      help="Aggregation strategy")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=256,
                      help="Maximum sequence length")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                      help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                      help="Checkpoint directory")
    parser.add_argument("--simulate_latency", action="store_true",
                      help="Simulate network latency")
    
    # Data arguments
    parser.add_argument("--data_distribution", type=str, default="iid",
                      choices=["iid", "non-iid"],
                      help="Data distribution across clients")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SimulationConfig(
        model_name=args.model_name,
        client_layers=args.client_layers,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        simulate_latency=args.simulate_latency,
        aggregation_strategy=args.aggregation,
        data_distribution=args.data_distribution
    )
    
    # Run simulation
    simulation = FederatedLearningSimulation(config)
    simulation.run()


if __name__ == "__main__":
    main()
