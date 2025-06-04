from model import LamineYamal
import torch
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

class HansiFlick:
    def __init__(self, input_shape=(3, 600, 800), num_actions=7, gamma=0.9, lr=1e-2, device='cuda'):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.model = LamineYamal(input_shape[0], input_shape[1], input_shape[2], num_actions).to(device)
        self.lr = lr
        self.optim = Adam(self.model.parameters(), lr=self.lr)
        
    def predict(self, frame_tensor):
        frame_tensor = frame_tensor.to(self.device)
        out = self.model(frame_tensor)
        probs = F.softmax(out)
        return probs
        
    def process_frame(self, frame_tensor):
        if frame_tensor.max() > 1.0:
            frame_tensor = frame_tensor / 255.0
        return frame_tensor
    
    def train_model(self, history):
        total_loss = 0
        steps = len(history['reward'])
        # for i in range(steps):
        #     for j in range(i+1, steps):
        #         history['reward'][i] += math.pow(self.gamma, j - i) * history['reward'][j]
        for i in range(steps):
            self.optim.zero_grad()
            probs = self.predict(history['state'][i])
            loss = -probs[0][history['action'][i]] * sum(history['reward'])
            total_loss += loss.item()
            loss.backward()
            self.optim.step()
        print(total_loss)
        print(sum(history['reward']))
