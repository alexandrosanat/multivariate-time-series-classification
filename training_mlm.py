from transformers import BertTokenizer, BertForMaskedLM
import torch
from tqdm import tqdm

#%%

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

#%%

with open("data/text_data_clean.txt", "r") as f:
    text = f.read().split("\n")

inputs = tokenizer(
    text, return_tensors="pt", max_length=512, truncation=True, padding="max_length"
)


#%%

# The label will be the same as the input
inputs["labels"] = inputs.input_ids.detach().clone()

rand = torch.rand(inputs.input_ids.shape)
mask_arr = (
    (rand < 0.15)
    * (inputs.input_ids != 101)
    * (inputs.input_ids != 102)
    * (inputs.input_ids != 0)
)

# Get the indexes that will be masked
selection = []

for i in range(inputs.input_ids.shape[0]):
    selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

# Replace index encodings with mask token
selection = []

for i in range(inputs.input_ids.shape[0]):
    selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())


#%%


class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


#%%

dataset = MeditationsDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

#%%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)
# activate training mode
model.train()
# initialize optimizer
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)


#%%

epochs = 2

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

#%%
