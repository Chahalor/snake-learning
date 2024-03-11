import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class Linear_Qnet(nn.Module) :
    def __init__(self, input_size:int, hidden_size:int, output_size:int) -> None:
        super().__init__()   
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x) :
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name:str="model.pth") -> None :
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path) :
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Qtrainer() :
    def __init__(self, lr:float, model, gamma:int) -> None:
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimize = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss
    
    def training_step(self, state, action, reward, next_state, done) -> None :
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1 :
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)

        target = pred.clone()
        for i in range(len(done)) :
            Q_new = reward[i]
            if not done[i] :
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            
            target[i][torch.argmax(action).item()] = Q_new 

        self.optimize.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimize.step()
