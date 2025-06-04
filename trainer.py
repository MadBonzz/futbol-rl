from model import LamineYamal
import torch
from torch.optim import Adam
from torch import nn
import numpy as np

class HansiFlick:
    def __init__(self, input_shape=(3, 600, 800), num_actions=7, epsilon = 1, epsilon_factor=0.995, lr=1e-2, device='cuda'):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device
        self.model = LamineYamal(input_shape[0], input_shape[1], input_shape[2], num_actions).to(device)
        self.eval_model = LamineYamal(input_shape[0], input_shape[1], input_shape[2], num_actions).to(device)
        self.eval_model.load_state_dict(self.model.state_dict())
        self.lr = lr
        self.optim = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.L1Loss()
        self.epsilon = epsilon
        self.decay_factory = epsilon_factor
        
    def predict(self, frame_tensor):
        prob = np.random.random()
        if prob <= self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            frame_tensor = frame_tensor.to(self.device)
            out = self.model(frame_tensor)
            return torch.argmax(out).item()
        
    def get_value(self, frame_tensor):
        frame_tensor = frame_tensor.to(self.device)
        out = self.model(frame_tensor)
        return out
    
    def get_target(self, frame_tensor):
        frame_tensor = frame_tensor.to(self.device)
        out = self.eval_model(frame_tensor)
        return out
        
    def process_frame(self, frame_tensor):
        if frame_tensor.max() > 1.0:
            frame_tensor = frame_tensor / 255.0
        return frame_tensor
    
    def train_model(self, history):
        total_loss = 0
        steps = len(history['state'])
        idxs = [i for i in range(steps)]
        idxs = np.random.choice(idxs, size=steps // 2)
        for idx in idxs:
            state = history['state'][idx]
            out = self.get_value(state)
            q_value = torch.max(out).to(self.device)
            target = history['reward'][idx]
            if history['next_state'][idx] is not None:
                target += torch.max(self.get_target(history['next_state'][idx]))
            target = torch.tensor(target, dtype=torch.float32).to(self.device)
            loss = self.criterion(q_value, target)
            total_loss += loss.item()
            loss.backward()
            self.optim.zero_grad()
            self.optim.step()
        print(total_loss)
        print(sum(history['reward']))
        self.eval_model.load_state_dict(self.model.state_dict())