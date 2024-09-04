import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

class ProgressiveNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, tasks):
        super(ProgressiveNeuralNetwork, self).__init__()
        self.columns = nn.ModuleList()  # List to hold different columns for tasks

        # Create initial column for the first task
        self.add_column(task_id=0)
        
        self.output_layers = nn.ModuleList([
            nn.Linear(512, output_dim) for _ in range(tasks)
        ])
    
    def forward_with_column(self, x, task_id, column_id):
        out = self.columns[column_id](x)
        return out

    def add_column(self, task_id):
        """Add a new column (series of layers) for a new task."""
        column = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.columns.append(column)

    def forward(self, x, task_id):
        out = self.columns[task_id](x)
        return self.output_layers[task_id](out)

def cosine_similarity(tensor1, tensor2):
    tensor1 = tensor1.flatten()
    tensor2 = tensor2.flatten()
    return torch.nn.functional.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()

def evaluate_column_performance(pnn, task_id, accumulated_test_data, accumulated_test_labels):
    performances = []
    for i in range(len(pnn.columns)):
        output = pnn.forward_with_column(accumulated_test_data, task_id, i)
        pred = output.argmax(dim=1, keepdim=True).squeeze()
        accuracy = accuracy_score(accumulated_test_labels.cpu(), pred.cpu())
        performances.append(accuracy)
    return performances

def remove_redundant_columns(pnn, accumulated_test_data, accumulated_test_labels, similarity_threshold=0.9):
    performances = evaluate_column_performance(pnn, len(pnn.columns) - 1, accumulated_test_data, accumulated_test_labels)
    to_remove = set()
    
    for i in range(len(pnn.columns)):
        for j in range(i + 1, len(pnn.columns)):
            similarity = cosine_similarity(pnn.columns[i].state_dict()['0.weight'], pnn.columns[j].state_dict()['0.weight'])
            if similarity >= similarity_threshold:
                if performances[i] < performances[j]:
                    to_remove.add(i)
                else:
                    to_remove.add(j)

    if to_remove:
        print(f"Removing {len(to_remove)} redundant columns based on similarity...")
        pnn.columns = nn.ModuleList([col for i, col in enumerate(pnn.columns) if i not in to_remove])
