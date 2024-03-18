import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
from Unet import Unet

class SegmentDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, train=True):
        self.transform = transform
        self.original_images = []
        self.segmented_images = []

        root = 'dataset'

        if train==True:
            original_image_path = os.path.join(root, 'train', 'original')
            segmented_image_path = os.path.join(root, 'train', 'segment')
        else:
            original_image_path = os.path.join(root, 'val', 'original')
            segmented_image_path = os.path.join(root, 'val', 'segment')

        original_image_files = os.listdir(original_image_path)
        segmented_image_files = os.listdir(segmented_image_path)

        for original_image_file in original_image_files:
            self.original_images.append(os.path.join(original_image_path, original_image_file))
            self.segmented_images.append(os.path.join(segmented_image_path, original_image_file.split('.')[0]+'.png'))


    def __getitem__(self, index):
        original = self.original_images[index]
        segmented = self.segmented_images[index]

        with open(original, 'rb') as f:
            image = Image.open(f)
            image = crop_to_square(image)
            image = resize_image(image, 512, antialias=True)
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        with open(segmented, 'rb') as f:
            label = Image.open(f)
            label = crop_to_square(label)
            label = resize_image(label, 512, antialias=True)
            label.convert('P')
            label = np.asarray(label)
            label = np.where(label == 255, 22-1, label)
            """
            segment_numpy = np.zeros([22, label.shape[0], label.shape[1]])
            for c in range(22):
                segment_numpy[c] = np.where(label==c, 1, 0)
            """
            #segment_numpy = torch.from_numpy(segment_numpy)

        return image, label

    def __len__(self):
        return len(self.original_images)
    
def crop_to_square(image):
    size = min(image.size)
    left, upper = (image.width - size) // 2, (image.height - size) // 2
    right, bottom = (image.width + size) // 2, (image.height + size) // 2
    return image.crop((left, upper, right, bottom))

def resize_image(image, to_size, antialias = False):
    if image.size != to_size:
        if antialias:
            image = image.resize((to_size, to_size), Image.LANCZOS)
        else:
            image = image.resize((to_size, to_size))
    return image

data_transforms = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = SegmentDataset(data_transforms, train=True)
test_dataset = SegmentDataset(data_transforms, train=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

#device = 'cuda' if torch.cuda.is_available else 'cpu'
device = 'cpu'
print("device : ", device)
net = Unet(3, 22, 64).to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_epochs = 100

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    train_loss = 0
    val_loss = 0
    
    #train
    net.train()
    with tqdm(train_loader, ncols=100) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels.long())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    
    #val
    net.eval()
    with torch.no_grad():
        with tqdm(test_loader, ncols=100) as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels.long())
                val_loss += loss.item()
    avg_val_loss = val_loss / len(test_loader.dataset)
    
    print ('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}' 
                   .format(epoch+1, num_epochs, loss=avg_train_loss, val_loss=avg_val_loss))
    train_loss_list.append(avg_train_loss)
    val_loss_list.append(avg_val_loss)

    if epoch % 10 ==0:
        torch.save(net.state_dict(), 'param/param-'+str(epoch)+'.pth')
    train_loss_list.append(avg_train_loss)
    val_loss_list.append(avg_val_loss)

with open('train_loss.pickle', 'wb') as f:
    pickle.dump(train_loss_list, f)
with open('val_loss.pickle', 'wb') as f:
    pickle.dump(val_loss_list, f)