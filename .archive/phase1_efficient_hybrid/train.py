"""Training script for teacher model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
from typing import Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teacher.models.efficient_hybrid import create_efficient_model
from teacher.loaders import create_dataloaders
from utils.vocabulary import Vocabulary, load_vocabulary_from_file
from utils.ctc import CTCLoss, ctc_decode, prepare_ctc_targets
from utils.metrics import compute_wer


class TeacherTrainer:
    """Trainer for teacher model."""
    
    def __init__(self, config: dict):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        
        # Paths - resolve relative to project root
        project_root = Path(__file__).parent.parent
        self.data_dir = (project_root / config['data']['data_dir']).resolve()
        self.features_dir = self.data_dir / config['data']['features_dir']
        self.annotations_dir = self.data_dir / config['data']['annotations_dir']
        self.vocab_file = self.data_dir / config['data']['vocab_file']
        self.checkpoint_dir = Path((project_root / config['training']['checkpoint_dir']).resolve())
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load vocabulary
        print(f"Loading vocabulary from {self.vocab_file}")
        self.vocabulary = load_vocabulary_from_file(self.vocab_file)
        print(f"Vocabulary size: {len(self.vocabulary)}")
        
        # Create data loaders
        print("Creating data loaders...")
        self.loaders = create_dataloaders(
            features_dir=self.features_dir,
            annotations_dir=self.annotations_dir,
            vocabulary=self.vocabulary,
            batch_size=config['training']['batch_size'],
            num_workers=max(config['training']['num_workers'], 0),
            max_length=config['training'].get('max_length'),
            normalize=config['training'].get('normalize', True)
        )
        
        # Create model
        print("Creating efficient hybrid model...")
        self.model = create_efficient_model(
            model_type='hybrid',
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model'].get('hidden_dim', 768),  # Default to 768 for better performance
            num_classes=len(self.vocabulary),
            dropout=config['model'].get('dropout', 0.3)
        )
        self.model = self.model.to(self.device)
        
        # Calculate model size and parameters
        num_params = self.model.count_parameters()
        model_size_mb = num_params * 4 / (1024 * 1024)  # Assuming float32
        print(f"Model parameters: {num_params:,}")
        print(f"Model size: {model_size_mb:.2f} MB")
        
        # Loss function with blank and repetition penalties to prevent CTC collapse
        blank_penalty = config['training'].get('blank_penalty', 0.0)
        repetition_penalty = config['training'].get('repetition_penalty', 0.0)
        self.criterion = CTCLoss(
            blank_idx=self.vocabulary.blank_idx,
            reduction='mean',
            blank_penalty=blank_penalty,
            repetition_penalty=repetition_penalty
        )
        
        # Optimizer
        learning_rate = config['training']['learning_rate']
        weight_decay = config['training'].get('weight_decay', 1e-5)
        # Ensure numeric types
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler - more aggressive to prevent overfitting
        # Use min_delta to avoid triggering on noise
        # Lower patience to reduce LR earlier when overfitting starts
        lr_patience = config['training'].get('lr_patience', 3)
        lr_min_delta = config['training'].get('lr_min_delta', 0.01)  # Minimum improvement required
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-6,  # Minimum learning rate
            threshold=lr_min_delta,  # Minimum change to qualify as improvement
            threshold_mode='rel',  # Relative threshold (percentage change)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_wer = float('inf')
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_wers = []
        self.learning_rates = []  # Track learning rate history
        self.train_val_gaps = []  # Track train-val gap for overfitting detection

        # Early stopping
        self.early_stop_patience = config['training'].get('early_stop_patience', 15)
        self.early_stop_counter = 0
        self.early_stop_min_delta = config['training'].get('early_stop_min_delta', 0.001)
        
        # Load checkpoint if specified
        if config['training'].get('resume_from'):
            self.load_checkpoint(config['training']['resume_from'])
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.loaders['train'], desc=f'Epoch {self.current_epoch+1}')
        for batch in pbar:
            features = batch['features'].to(self.device)
            sequence_lengths = batch['sequence_lengths'].to(self.device)
            targets = batch['targets']
            target_lengths = batch['target_lengths']

            # Prepare CTC targets
            targets_list = [targets[i, :target_lengths[i]].tolist()
                          for i in range(len(targets))]
            targets_tensor, target_lengths_tensor = prepare_ctc_targets(
                targets_list, self.device
            )

            # Forward pass
            self.optimizer.zero_grad()
            log_probs = self.model(features, sequence_lengths)

            # Model preserves full temporal resolution (no downsampling)
            # CTC loss expects input_lengths to match log_probs length T
            # Sequence lengths stay the same since there's no downsampling
            T = log_probs.shape[0]
            sequence_lengths_clamped = torch.clamp(sequence_lengths, max=T)

            # Compute loss
            loss = self.criterion(
                log_probs,
                targets_tensor,
                sequence_lengths_clamped,
                target_lengths_tensor
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training'].get('grad_clip', 5.0)
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

            # Clear CUDA cache periodically (every 100 batches) to prevent memory buildup
            if num_batches % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches

        # Clear CUDA cache after epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss
    
    def validate(self) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_references = []
        all_hypotheses = []

        with torch.no_grad():
            for batch in tqdm(self.loaders['dev'], desc='Validation'):
                features = batch['features'].to(self.device)
                sequence_lengths = batch['sequence_lengths'].to(self.device)
                targets = batch['targets']
                target_lengths = batch['target_lengths']

                # Forward pass
                log_probs = self.model(features, sequence_lengths)

                # Prepare CTC targets
                targets_list = [targets[i, :target_lengths[i]].tolist()
                              for i in range(len(targets))]
                targets_tensor, target_lengths_tensor = prepare_ctc_targets(
                    targets_list, self.device
                )

                # Model preserves full temporal resolution (no downsampling)
                # CTC loss expects input_lengths to match log_probs length T
                T = log_probs.shape[0]
                sequence_lengths_clamped = torch.clamp(sequence_lengths, max=T)

                # Compute loss
                loss = self.criterion(
                    log_probs,
                    targets_tensor,
                    sequence_lengths_clamped,
                    target_lengths_tensor
                )
                total_loss += loss.item()
                num_batches += 1

                # Decode predictions with beam search (more robust than greedy)
                # Note: log_probs is [T, N, C], ctc_decode expects [T, N, C]
                decode_method = 'greedy'
                predictions = ctc_decode(log_probs,
                                       blank_idx=self.vocabulary.blank_idx,
                                       method=decode_method,
                                       beam_width=20)

                # Decode references and hypotheses
                for i, pred in enumerate(predictions):
                    hyp_text = self.vocabulary.decode(pred)
                    ref_text = batch['annotations'][i]
                    all_references.append(ref_text)
                    all_hypotheses.append(hyp_text)

        avg_loss = total_loss / num_batches
        wer, wer_stats = compute_wer(all_references, all_hypotheses)

        # Clear CUDA cache after validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, wer
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_wer': self.best_wer,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_wers': self.val_wers,
            'config': self.config
        }

        # Save latest
        checkpoint_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with WER: {self.best_wer*100:.2f}%")

        # Save epoch checkpoint every 10 epochs (reduce disk usage for long training)
        if self.current_epoch % 10 == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{self.current_epoch}.pt'
            torch.save(checkpoint, epoch_path)

        # Clear checkpoint dict from memory and clear CUDA cache
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_wer = checkpoint['best_wer']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_wers = checkpoint.get('val_wers', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        self.train_val_gaps = checkpoint.get('train_val_gaps', [])
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Training samples: {len(self.loaders['train'].dataset)}")
        print(f"Validation samples: {len(self.loaders['dev'].dataset)}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Number of workers: {self.config['training']['num_workers']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")


        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_wer = self.validate()
            self.val_losses.append(val_loss)
            self.val_wers.append(val_wer)

            # Track learning rate BEFORE scheduler step
            current_lr_before = self.optimizer.param_groups[0]['lr']
            
            # Update learning rate based on validation LOSS
            # The scheduler will reduce LR if val_loss doesn't improve by min_delta
            # within patience epochs
            self.scheduler.step(val_loss)
            
            # Track learning rate AFTER scheduler step
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Track train-val gap for overfitting detection
            train_val_gap = train_loss - val_loss
            self.train_val_gaps.append(train_val_gap)

            # Early stopping check (based on validation loss)
            if val_loss < self.best_val_loss - self.early_stop_min_delta:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            # Print progress
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val WER: {val_wer*100:.2f}%")
            print(f"LR: {current_lr:.6f}")
            print(f"Early Stop Counter: {self.early_stop_counter}/{self.early_stop_patience}")

            # Save checkpoint (based on WER for best model selection)
            is_best = val_wer < self.best_wer
            if is_best:
                self.best_wer = val_wer

            self.save_checkpoint(is_best)

            # Check early stopping BEFORE continuing to next epoch
            if self.early_stop_counter >= self.early_stop_patience:
                print(f"\nEarly stopping triggered!")
                print(f"   No improvement for {self.early_stop_patience} epochs")
                print(f"   Best Val Loss: {self.best_val_loss:.4f}")
                print(f"   Best WER: {self.best_wer*100:.2f}%")
                break

            # Save training history
            history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_wers': self.val_wers,
                'learning_rates': self.learning_rates,
                'train_val_gaps': self.train_val_gaps
            }
            with open(self.checkpoint_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)
            
            # Plot training curves
            self.plot_training_curves()

            # Clear CUDA cache at the end of each epoch to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\nTraining completed! Best WER: {self.best_wer*100:.2f}%")
    
    def plot_training_curves(self):
        """Generate publication-quality plots for thesis."""
        epochs = range(1, len(self.train_losses) + 1)
        
        # Set style for thesis-quality plots
        try:
            plt.style.use('seaborn-v0_8-paper')
        except:
            try:
                plt.style.use('seaborn-paper')
            except:
                plt.style.use('seaborn-whitegrid')
        sns.set_palette("husl")
        fig_width = 12  # Good for thesis (2-column layout)
        
        # Create figure with subplots (2x2 layout)
        fig, axes = plt.subplots(2, 2, figsize=(fig_width, 8))
        fig.suptitle('Teacher Model Training Progress', fontsize=16, fontweight='bold')
        
        # 1. Loss curves (train vs validation)
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        
        # 2. WER over epochs
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.val_wers, 'g-', linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=self.best_wer, color='r', linestyle='--', 
                   label=f'Best WER: {self.best_wer*100:.2f}%', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Word Error Rate (%)', fontsize=11)
        ax2.set_title('Validation Word Error Rate', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        
        # 3. Learning rate schedule
        ax3 = axes[1, 0]
        if len(self.learning_rates) > 0:
            ax3.plot(epochs, self.learning_rates[:len(epochs)], 'm-', linewidth=2, marker='s', markersize=4)
            ax3.set_xlabel('Epoch', fontsize=11)
            ax3.set_ylabel('Learning Rate', fontsize=11)
            ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(left=0)
        else:
            ax3.text(0.5, 0.5, 'No LR data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        
        # 4. Combined metrics overview
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()
        
        # Loss on left axis
        line1 = ax4.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)
        line2 = ax4.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2, alpha=0.7)
        
        # WER on right axis
        line3 = ax4_twin.plot(epochs, self.val_wers, 'g-', label='WER (%)', 
                              linewidth=2, marker='o', markersize=4)
        
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Loss', fontsize=11, color='black')
        ax4_twin.set_ylabel('WER (%)', fontsize=11, color='green')
        ax4.set_title('Training Overview', fontsize=12, fontweight='bold')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(left=0)
        
        plt.tight_layout()
        
        # Save high-resolution plot for thesis
        plot_path = self.checkpoint_dir / 'training_curves.pdf'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Saved training curves: {plot_path}")
        
        # Also save as PNG for quick viewing
        plot_path_png = self.checkpoint_dir / 'training_curves.png'
        plt.savefig(plot_path_png, dpi=300, bbox_inches='tight', format='png')
        
        plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train teacher model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and train
    trainer = TeacherTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

